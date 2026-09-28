//! Native regex and entropy scanner for Harpocrates.
//!
//! The crate deliberately stops at candidate generation. Python remains the
//! owner of classification, context feature extraction, and ML verification.

use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{Read, Take};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::thread;

pub const PROTOCOL_VERSION: u32 = 1;

const HIGH_RISK_EXTENSIONS: &[&str] = &[".env", ".pem", ".key", ".secret", ".credentials"];
const LOW_NOISE_EXTENSIONS: &[&str] = &[".html", ".htm", ".css", ".scss", ".sass", ".svg", ".xml"];

#[derive(Debug)]
enum IgnorePattern {
    Exact(String),
    Glob(Regex),
}

impl IgnorePattern {
    fn compile(pattern: &str) -> Result<Self, ScanError> {
        if pattern.contains(['*', '?', '[']) {
            let expression = glob_regex(pattern);
            Regex::new(&expression)
                .map(Self::Glob)
                .map_err(|error| ScanError::new(format!("invalid ignore pattern: {error}")))
        } else {
            #[cfg(windows)]
            let pattern = pattern.to_lowercase();
            #[cfg(not(windows))]
            let pattern = pattern.to_owned();
            Ok(Self::Exact(pattern))
        }
    }

    fn matches(&self, name: &str) -> bool {
        #[cfg(windows)]
        let name = name.to_lowercase();
        #[cfg(not(windows))]
        let name = name.to_owned();
        match self {
            Self::Exact(pattern) => pattern == &name,
            Self::Glob(pattern) => pattern.is_match(&name),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Evidence {
    Regex,
    Entropy,
    Ml,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawFinding {
    #[serde(rename = "type")]
    pub kind: String,
    pub line: usize,
    pub snippet: String,
    pub token: String,
    pub token_start: Option<usize>,
    pub token_end: Option<usize>,
    pub evidence: Evidence,
    pub severity: String,
    pub confidence: f64,
    pub entropy: f64,
    pub in_comment: Option<bool>,
    pub var_name: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ScanOptions {
    pub file: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileScanRequest {
    pub path: PathBuf,
    pub max_bytes: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchScanRequest {
    pub protocol_version: u32,
    pub files: Vec<FileScanRequest>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FileScanResponse {
    pub path: PathBuf,
    pub findings: Vec<RawFinding>,
    pub scanned: bool,
    pub line_count: usize,
    pub bytes_scanned: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchScanResponse {
    pub protocol_version: u32,
    pub files: Vec<FileScanResponse>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryScanRequest {
    pub protocol_version: u32,
    pub root: PathBuf,
    pub recursive: bool,
    pub max_file_size: u64,
    pub ignore_patterns: Vec<String>,
}

#[derive(Debug)]
pub struct ScanError(String);

impl ScanError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ScanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ScanError {}

struct Signature {
    name: &'static str,
    pattern: Regex,
}

struct Engine {
    critical: Vec<Signature>,
    critical_set: RegexSet,
    high: Vec<Signature>,
    high_set: RegexSet,
    token: Regex,
    url: Regex,
    pem_body: Regex,
    comment_strip: Regex,
    prose_filter: Regex,
    sensitive_assignment: Regex,
    env_assignment: Regex,
}

impl Engine {
    fn compile() -> Result<Self, regex::Error> {
        let critical_definitions = [
            (
                "AWS_ACCESS_KEY_ID",
                r"\b(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b",
            ),
            (
                "GITHUB_PAT",
                r"\b(?:gh[pousr]_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59})\b",
            ),
            (
                "SLACK_TOKEN",
                r"\bxox[pboar]-[0-9]{10,13}-[a-zA-Z0-9\-]{24,34}\b",
            ),
            ("STRIPE_KEY", r"\b[rs]k_(?:live|test)_[a-zA-Z0-9]{24,99}\b"),
            (
                "OPENAI_API_KEY",
                r"\b(?:sk-[a-zA-Z0-9]{48}|sk-proj-[a-zA-Z0-9]{20}T3[a-zA-Z0-9]{20,})\b",
            ),
            ("ANTHROPIC_API_KEY", r"\bsk-ant-api03-[a-zA-Z0-9\-_]{93,}\b"),
            ("GCP_API_KEY", r"\bAIza[0-9A-Za-z\-_]{35}\b"),
            ("NPM_TOKEN", r"\bnpm_[a-zA-Z0-9]{36}\b"),
            ("PYPI_TOKEN", r"\bpypi-[a-zA-Z0-9\-_]{50,}\b"),
        ];
        let high_definitions = [
            (
                "PRIVATE_KEY",
                r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            ),
            (
                "SLACK_WEBHOOK",
                r"\bhttps://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,10}/B[a-zA-Z0-9_]{8,10}/[a-zA-Z0-9_]{24}\b",
            ),
            (
                "DISCORD_WEBHOOK",
                r"\bhttps://discord\.com/api/webhooks/[0-9]{17,19}/[a-zA-Z0-9\-_]{68}\b",
            ),
            (
                "SENDGRID_API_KEY",
                r"\bSG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}\b",
            ),
            ("TWILIO_API_KEY", r"\bSK[0-9a-fA-F]{32}\b"),
            ("DATABRICKS_TOKEN", r"\bdapi[a-f0-9]{32}\b"),
            ("HASHICORP_VAULT_TOKEN", r"\bhvs\.[a-zA-Z0-9\-_]{90,}\b"),
            ("OPENAI_API_KEY_LEGACY", r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        ];
        Ok(Self {
            critical: compile_signatures(&critical_definitions)?,
            critical_set: RegexSet::new(critical_definitions.map(|(_, pattern)| pattern))?,
            high: compile_signatures(&high_definitions)?,
            high_set: RegexSet::new(high_definitions.map(|(_, pattern)| pattern))?,
            token: Regex::new(r"[A-Za-z0-9+/=_\-]{20,}")?,
            url: Regex::new(r"(?:https?://|data:)\S+")?,
            pem_body: Regex::new(r"^[A-Za-z0-9+/]{64}$")?,
            comment_strip: Regex::new(r"^(?:#+|//+|/\*+|\*+|<!--|--)\s*")?,
            prose_filter: Regex::new(r#"[=:'\"]"#)?,
            sensitive_assignment: Regex::new(
                r#"(?i)(?:^|[^A-Za-z])((?:pass(?:word|wd|w)?|pwd|usr(?:name)?|user|host|conn(?:ection|str)?|secret|token|key|auth|cred)[a-z0-9_]*)\s*[:=]\s*['\"]([^'\"]{3,100})['\"]"#,
            )?,
            env_assignment: Regex::new(
                r#"(?i)(?:^|[^A-Za-z])((?:pass(?:word|wd|w)?|pwd|conn(?:ection|str)?|secret|token|key|auth|cred|url|endpoint|callback)[a-z0-9_]*)\s*=\s*([^\s'\"]{3,200})"#,
            )?,
        })
    }
}

fn compile_signatures(
    definitions: &[(&'static str, &'static str)],
) -> Result<Vec<Signature>, regex::Error> {
    definitions
        .iter()
        .map(|(name, pattern)| {
            Ok(Signature {
                name,
                pattern: Regex::new(pattern)?,
            })
        })
        .collect()
}

static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();

fn engine() -> Result<&'static Engine, ScanError> {
    ENGINE
        .get_or_init(|| Engine::compile().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|message| ScanError::new(format!("failed to compile scanner regexes: {message}")))
}

pub fn scan_text(text: &str, options: &ScanOptions) -> Result<Vec<RawFinding>, ScanError> {
    let engine = engine()?;
    let extension = effective_extension(options.file.as_deref());
    let mut findings = Vec::new();
    let mut in_block_comment = false;

    for (index, line) in text.lines().enumerate() {
        let stripped = line.trim();
        let opens = stripped.matches("/*").count();
        let closes = stripped.matches("*/").count();
        let was_in_block = in_block_comment && !stripped.starts_with("/*");
        let mut line_findings = scan_line(engine, line, index + 1, &extension);
        if was_in_block {
            for finding in &mut line_findings {
                if finding.in_comment.is_none() {
                    finding.in_comment = Some(true);
                }
            }
        }
        findings.extend(line_findings);
        in_block_comment = (in_block_comment && closes == 0) || opens > closes;
    }

    Ok(findings)
}

pub fn scan_file(
    path: impl AsRef<Path>,
    max_bytes: Option<usize>,
) -> Result<Vec<RawFinding>, ScanError> {
    scan_file_response(path.as_ref(), max_bytes).map(|response| response.findings)
}

pub fn scan_batch(request: BatchScanRequest) -> Result<BatchScanResponse, ScanError> {
    if request.protocol_version != PROTOCOL_VERSION {
        return Err(ScanError::new(format!(
            "unsupported protocol version {}; expected {PROTOCOL_VERSION}",
            request.protocol_version
        )));
    }

    let files = request
        .files
        .into_iter()
        .map(
            |file| match scan_file_response(&file.path, file.max_bytes) {
                Ok(response) => response,
                Err(error) => FileScanResponse {
                    path: file.path,
                    findings: Vec::new(),
                    scanned: false,
                    line_count: 0,
                    bytes_scanned: 0,
                    error: Some(error.to_string()),
                },
            },
        )
        .collect();

    Ok(BatchScanResponse {
        protocol_version: PROTOCOL_VERSION,
        files,
    })
}

pub fn scan_directory(request: DirectoryScanRequest) -> Result<BatchScanResponse, ScanError> {
    if request.protocol_version != PROTOCOL_VERSION {
        return Err(ScanError::new(format!(
            "unsupported protocol version {}; expected {PROTOCOL_VERSION}",
            request.protocol_version
        )));
    }
    if !request.root.is_dir() {
        return Err(ScanError::new(format!(
            "not a directory: {}",
            request.root.display()
        )));
    }

    let ignore_patterns = request
        .ignore_patterns
        .iter()
        .map(|pattern| IgnorePattern::compile(pattern))
        .collect::<Result<Vec<_>, _>>()?;
    let mut paths = Vec::new();
    collect_files(
        &request.root,
        request.recursive,
        &ignore_patterns,
        &mut paths,
    )?;
    paths.sort();

    let files = scan_directory_paths(&paths, request.max_file_size)?;

    Ok(BatchScanResponse {
        protocol_version: PROTOCOL_VERSION,
        files,
    })
}

fn scan_directory_paths(
    paths: &[PathBuf],
    max_file_size: u64,
) -> Result<Vec<FileScanResponse>, ScanError> {
    let workers = configured_worker_count(paths.len());
    if workers <= 1 {
        return Ok(paths
            .iter()
            .cloned()
            .map(|path| scan_path_with_limit(path, max_file_size))
            .collect());
    }

    let chunk_size = paths.len().div_ceil(workers);
    thread::scope(|scope| {
        let handles: Vec<_> = paths
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .cloned()
                        .map(|path| scan_path_with_limit(path, max_file_size))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut files = Vec::with_capacity(paths.len());
        for handle in handles {
            files.extend(
                handle
                    .join()
                    .map_err(|_| ScanError::new("native scan worker panicked"))?,
            );
        }
        Ok(files)
    })
}

fn configured_worker_count(task_count: usize) -> usize {
    if task_count < 32 {
        return 1;
    }
    let available = thread::available_parallelism().map_or(1, usize::from);
    let configured = std::env::var("HARPOCRATES_RUST_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(available);
    configured.min(available).min(8).min(task_count)
}

fn scan_path_with_limit(path: PathBuf, max_file_size: u64) -> FileScanResponse {
    let metadata = fs::metadata(&path)
        .map_err(|error| ScanError::new(format!("failed to inspect {}: {error}", path.display())));
    match metadata {
        Ok(metadata) if metadata.len() > max_file_size => FileScanResponse {
            path,
            findings: Vec::new(),
            scanned: false,
            line_count: 0,
            bytes_scanned: 0,
            error: Some(format!(
                "skipped large file ({} bytes exceeds {} byte limit)",
                metadata.len(),
                max_file_size
            )),
        },
        Ok(_) => match scan_file_response(&path, Some(max_file_size as usize)) {
            Ok(response) => response,
            Err(error) => FileScanResponse {
                path,
                findings: Vec::new(),
                scanned: false,
                line_count: 0,
                bytes_scanned: 0,
                error: Some(error.to_string()),
            },
        },
        Err(error) => FileScanResponse {
            path,
            findings: Vec::new(),
            scanned: false,
            line_count: 0,
            bytes_scanned: 0,
            error: Some(error.to_string()),
        },
    }
}

fn collect_files(
    directory: &Path,
    recursive: bool,
    ignore_patterns: &[IgnorePattern],
    paths: &mut Vec<PathBuf>,
) -> Result<(), ScanError> {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .map_err(|error| {
            ScanError::new(format!(
                "failed to read directory {}: {error}",
                directory.display()
            ))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            ScanError::new(format!(
                "failed to read directory {}: {error}",
                directory.display()
            ))
        })?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| {
            ScanError::new(format!("failed to inspect {}: {error}", path.display()))
        })?;
        if file_type.is_symlink() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if ignore_patterns.iter().any(|pattern| pattern.matches(&name)) {
            continue;
        }
        if file_type.is_dir() {
            if recursive {
                collect_files(&path, true, ignore_patterns, paths)?;
            }
        } else if file_type.is_file() {
            paths.push(path);
        }
    }
    Ok(())
}

fn glob_regex(pattern: &str) -> String {
    let characters: Vec<char> = pattern.chars().collect();
    let mut expression = String::from("^");
    #[cfg(windows)]
    expression.push_str("(?i)");
    let mut index = 0;
    while index < characters.len() {
        match characters[index] {
            '*' => expression.push_str(".*"),
            '?' => expression.push('.'),
            '[' => {
                let mut end = index + 1;
                if end < characters.len() && matches!(characters[end], '!' | '^') {
                    end += 1;
                }
                if end < characters.len() && characters[end] == ']' {
                    end += 1;
                }
                while end < characters.len() && characters[end] != ']' {
                    end += 1;
                }
                if end == characters.len() {
                    expression.push_str(r"\[");
                } else {
                    expression.push('[');
                    let mut class_index = index + 1;
                    if characters[class_index] == '!' {
                        expression.push('^');
                        class_index += 1;
                    } else if characters[class_index] == '^' {
                        expression.push_str(r"\^");
                        class_index += 1;
                    }
                    for character in &characters[class_index..end] {
                        if matches!(character, '\\' | '[') {
                            expression.push('\\');
                        }
                        expression.push(*character);
                    }
                    expression.push(']');
                    index = end;
                }
            }
            literal => expression.push_str(&regex::escape(&literal.to_string())),
        }
        index += 1;
    }
    expression.push('$');
    expression
}

fn scan_file_response(
    path: &Path,
    max_bytes: Option<usize>,
) -> Result<FileScanResponse, ScanError> {
    let bytes = read_bounded(path, max_bytes)?;
    let sample_len = bytes.len().min(1024);
    if looks_binary(&bytes[..sample_len]) {
        return Ok(FileScanResponse {
            path: path.to_path_buf(),
            findings: Vec::new(),
            scanned: true,
            line_count: 0,
            bytes_scanned: bytes.len(),
            error: None,
        });
    }

    let content = String::from_utf8_lossy(&bytes);
    let findings = scan_text(
        &content,
        &ScanOptions {
            file: Some(path.to_string_lossy().into_owned()),
        },
    )?;
    Ok(FileScanResponse {
        path: path.to_path_buf(),
        findings,
        scanned: true,
        line_count: content.lines().count(),
        bytes_scanned: bytes.len(),
        error: None,
    })
}

fn read_bounded(path: &Path, max_bytes: Option<usize>) -> Result<Vec<u8>, ScanError> {
    let file = File::open(path)
        .map_err(|error| ScanError::new(format!("failed to read {}: {error}", path.display())))?;
    let mut bytes = Vec::new();
    match max_bytes {
        Some(limit) => {
            let mut reader: Take<File> = file.take(limit.saturating_add(1) as u64);
            reader.read_to_end(&mut bytes).map_err(|error| {
                ScanError::new(format!("failed to read {}: {error}", path.display()))
            })?;
            if bytes.len() > limit {
                bytes.truncate(limit);
                match bytes.iter().rposition(|byte| *byte == b'\n') {
                    Some(last_newline) => bytes.truncate(last_newline + 1),
                    None => bytes.clear(),
                }
            }
        }
        None => {
            let mut reader = file;
            reader.read_to_end(&mut bytes).map_err(|error| {
                ScanError::new(format!("failed to read {}: {error}", path.display()))
            })?;
        }
    }
    Ok(bytes)
}

fn looks_binary(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    if sample.contains(&0) {
        return true;
    }
    match std::str::from_utf8(sample) {
        Ok(_) => return false,
        Err(error) if error.error_len().is_none() => return false,
        Err(_) => {}
    }
    let non_text = sample
        .iter()
        .filter(|byte| {
            !matches!(byte, b'\t' | b'\n' | b'\r' | 0x0c | 0x08) && (**byte < 32 || **byte > 126)
        })
        .count();
    non_text as f64 / sample.len() as f64 > 0.3
}

fn scan_line(engine: &Engine, line: &str, lineno: usize, extension: &str) -> Vec<RawFinding> {
    let stripped = line.trim();
    if stripped.is_empty() {
        return Vec::new();
    }

    // PEM headers begin with five hyphens; they are not SQL comments.
    let is_comment = !stripped.starts_with("-----BEGIN ")
        && ["#", "//", "/*", "*", "<!--", "--"]
            .iter()
            .any(|prefix| stripped.starts_with(prefix));
    let scan_target: Cow<'_, str> = if is_comment {
        engine.comment_strip.replace(stripped, "")
    } else {
        Cow::Borrowed(stripped)
    };
    let comment_flag = is_comment.then_some(true);
    let snippet: String = stripped.chars().take(200).collect();
    let mut findings = Vec::new();

    for (signatures, signature_set, severity, confidence) in [
        (&engine.critical, &engine.critical_set, "critical", 0.99),
        (&engine.high, &engine.high_set, "high", 0.95),
    ] {
        for signature_index in signature_set.matches(&scan_target).iter() {
            let signature = &signatures[signature_index];
            for match_ in signature.pattern.find_iter(&scan_target) {
                let token = match_.as_str();
                if signature_is_excluded(signature.name, token) {
                    continue;
                }
                findings.push(RawFinding {
                    kind: signature.name.to_owned(),
                    line: lineno,
                    snippet: snippet.clone(),
                    token: token.to_owned(),
                    token_start: Some(byte_to_char_index(&scan_target, match_.start())),
                    token_end: Some(byte_to_char_index(&scan_target, match_.end())),
                    evidence: Evidence::Regex,
                    severity: severity.to_owned(),
                    confidence,
                    entropy: shannon_entropy(token),
                    in_comment: comment_flag,
                    var_name: None,
                });
            }
        }
    }

    if !findings.is_empty() {
        return findings;
    }
    if is_comment && !engine.prose_filter.is_match(&scan_target) {
        return findings;
    }
    if engine.pem_body.is_match(&scan_target) {
        return findings;
    }

    if LOW_NOISE_EXTENSIONS.contains(&extension) {
        return findings;
    }
    let scan_text: Cow<'_, str> = if HIGH_RISK_EXTENSIONS.contains(&extension) {
        Cow::Borrowed(&scan_target)
    } else {
        engine.url.replace_all(&scan_target, " ")
    };

    for match_ in engine.token.find_iter(&scan_text) {
        let token = match_.as_str();
        if let Some(entropy) = secret_entropy(token, 4.0) {
            findings.push(RawFinding {
                kind: "ENTROPY_CANDIDATE".to_owned(),
                line: lineno,
                snippet: snippet.clone(),
                token: token.to_owned(),
                token_start: None,
                token_end: None,
                evidence: Evidence::Entropy,
                severity: "info".to_owned(),
                confidence: entropy_confidence(entropy),
                entropy,
                in_comment: comment_flag,
                var_name: extract_var_name(&scan_target, token),
            });
        }
    }

    let mut found_tokens: HashSet<String> = findings
        .iter()
        .map(|finding| finding.token.clone())
        .collect();
    for captures in engine.sensitive_assignment.captures_iter(&scan_text) {
        let Some(var_match) = captures.get(1) else {
            continue;
        };
        let Some(value_match) = captures.get(2) else {
            continue;
        };
        if found_tokens.insert(value_match.as_str().to_owned()) {
            findings.push(candidate_finding(
                "ML_CANDIDATE",
                Evidence::Ml,
                0.5,
                lineno,
                &snippet,
                &scan_text,
                value_match,
                var_match.as_str(),
                comment_flag,
            ));
        }
    }

    if HIGH_RISK_EXTENSIONS.contains(&extension) {
        for captures in engine.env_assignment.captures_iter(&scan_text) {
            let Some(var_match) = captures.get(1) else {
                continue;
            };
            let Some(value_match) = captures.get(2) else {
                continue;
            };
            if found_tokens.insert(value_match.as_str().to_owned()) {
                findings.push(candidate_finding(
                    "ENV_ASSIGNMENT",
                    Evidence::Regex,
                    0.5,
                    lineno,
                    &snippet,
                    &scan_text,
                    value_match,
                    var_match.as_str(),
                    comment_flag,
                ));
            }
        }
    }

    findings
}

#[allow(clippy::too_many_arguments)]
fn candidate_finding(
    kind: &str,
    evidence: Evidence,
    confidence: f64,
    line: usize,
    snippet: &str,
    scan_text: &str,
    value_match: regex::Match<'_>,
    var_name: &str,
    in_comment: Option<bool>,
) -> RawFinding {
    let token = value_match.as_str();
    RawFinding {
        kind: kind.to_owned(),
        line,
        snippet: snippet.to_owned(),
        token: token.to_owned(),
        token_start: Some(byte_to_char_index(scan_text, value_match.start())),
        token_end: Some(byte_to_char_index(scan_text, value_match.end())),
        evidence,
        severity: "info".to_owned(),
        confidence,
        entropy: shannon_entropy(token),
        in_comment,
        var_name: Some(var_name.to_owned()),
    }
}

fn signature_is_excluded(name: &str, token: &str) -> bool {
    if name != "OPENAI_API_KEY_LEGACY" {
        return false;
    }
    let Some(body) = token.strip_prefix("sk-") else {
        return false;
    };
    body.starts_with("proj-")
        || body.starts_with("ant-api03-")
        || (body.len() == 48 && body.bytes().all(|byte| byte.is_ascii_alphanumeric()))
}

fn effective_extension(file: Option<&str>) -> String {
    let Some(file) = file else {
        return String::new();
    };
    let path = PathBuf::from(file);
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if basename == ".env" || basename.starts_with(".env.") {
        return ".env".to_owned();
    }
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!(".{}", extension.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn extract_var_name(line: &str, token: &str) -> Option<String> {
    let token_start = line.find(token)?;
    let prefix = line[..token_start].trim_end();
    let assignment = prefix.rsplit_once(['=', ':'])?.0.trim_end();
    let name: String = assignment
        .chars()
        .rev()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    (!name.is_empty()).then_some(name)
}

fn byte_to_char_index(value: &str, byte_index: usize) -> usize {
    value[..byte_index].chars().count()
}

fn entropy_confidence(entropy: f64) -> f64 {
    if entropy >= 5.5 {
        0.8
    } else if entropy >= 4.0 {
        0.6 + (entropy - 4.0) * (0.2 / 1.5)
    } else {
        0.6
    }
}

fn secret_entropy(value: &str, threshold: f64) -> Option<f64> {
    if value.len() < 20 {
        return None;
    }
    let mut seen = [false; 256];
    let mut unique = 0;
    let mut has_upper = false;
    let mut has_lower = false;
    let mut has_digit = false;
    let mut has_special = false;
    for byte in value.bytes() {
        let slot = &mut seen[usize::from(byte)];
        if !*slot {
            *slot = true;
            unique += 1;
        }
        has_upper |= byte.is_ascii_uppercase();
        has_lower |= byte.is_ascii_lowercase();
        has_digit |= byte.is_ascii_digit();
        has_special |= !byte.is_ascii_alphanumeric();
    }
    if unique <= 3 {
        return None;
    }
    if [has_upper, has_lower, has_digit, has_special]
        .iter()
        .filter(|present| **present)
        .count()
        < 2
    {
        return None;
    }
    if !has_digit && !has_special {
        return None;
    }
    if has_lower && !has_upper && !has_digit {
        return None;
    }
    let entropy = shannon_entropy(value);
    (entropy >= threshold).then_some(entropy)
}

fn shannon_entropy(value: &str) -> f64 {
    if value.is_empty() {
        return 0.0;
    }
    if value.is_ascii() {
        let mut counts = [0_u32; 256];
        for byte in value.bytes() {
            counts[usize::from(byte)] += 1;
        }
        let length = value.len() as f64;
        return counts
            .iter()
            .filter(|count| **count > 0)
            .map(|count| {
                let probability = f64::from(*count) / length;
                -probability * probability.log2()
            })
            .sum();
    }

    let mut counts = HashMap::new();
    for character in value.chars() {
        *counts.entry(character).or_insert(0usize) += 1;
    }
    let length = value.chars().count() as f64;
    counts
        .values()
        .map(|count| {
            let probability = *count as f64 / length;
            -probability * probability.log2()
        })
        .sum()
}
