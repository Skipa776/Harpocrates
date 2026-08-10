//! Native regex and entropy scanner for Harpocrates.
//!
//! The crate deliberately stops at candidate generation. Python remains the
//! owner of classification, context feature extraction, and ML verification.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const HIGH_RISK_EXTENSIONS: &[&str] = &[".env", ".pem", ".key", ".secret", ".credentials"];
const LOW_NOISE_EXTENSIONS: &[&str] = &[".html", ".htm", ".css", ".scss", ".sass", ".svg", ".xml"];

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
    high: Vec<Signature>,
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
        Ok(Self {
            critical: compile_signatures(&[
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
            ])?,
            high: compile_signatures(&[
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
                // Rust's regex engine intentionally omits look-around. The
                // Python exclusions are applied in `signature_is_excluded`.
                ("OPENAI_API_KEY_LEGACY", r"\bsk-[A-Za-z0-9_-]{16,}\b"),
            ])?,
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
    let mut findings = Vec::new();
    let mut in_block_comment = false;

    for (index, line) in text.lines().enumerate() {
        let stripped = line.trim();
        let opens = stripped.matches("/*").count();
        let closes = stripped.matches("*/").count();
        let was_in_block = in_block_comment && !stripped.starts_with("/*");
        let mut line_findings = scan_line(engine, line, index + 1, options.file.as_deref());
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
    let path = path.as_ref();
    let bytes = fs::read(path)
        .map_err(|error| ScanError::new(format!("failed to read {}: {error}", path.display())))?;
    let sample_len = bytes.len().min(1024);
    if looks_binary(&bytes[..sample_len]) {
        return Ok(Vec::new());
    }

    let bytes = bounded_complete_lines(&bytes, max_bytes);
    let content = String::from_utf8_lossy(bytes);
    scan_text(
        &content,
        &ScanOptions {
            file: Some(path.to_string_lossy().into_owned()),
        },
    )
}

fn bounded_complete_lines(bytes: &[u8], max_bytes: Option<usize>) -> &[u8] {
    let Some(limit) = max_bytes else {
        return bytes;
    };
    if bytes.len() <= limit {
        return bytes;
    }
    let prefix = &bytes[..limit];
    match prefix.iter().rposition(|byte| *byte == b'\n') {
        Some(last_newline) => &prefix[..=last_newline],
        None => &[],
    }
}

fn looks_binary(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    if sample.contains(&0) {
        return true;
    }
    let non_text = sample
        .iter()
        .filter(|byte| {
            !matches!(byte, b'\t' | b'\n' | b'\r' | 0x0c | 0x08) && (**byte < 32 || **byte > 126)
        })
        .count();
    non_text as f64 / sample.len() as f64 > 0.3
}

fn scan_line(engine: &Engine, line: &str, lineno: usize, file: Option<&str>) -> Vec<RawFinding> {
    let stripped = line.trim();
    if stripped.is_empty() {
        return Vec::new();
    }

    // PEM headers begin with five hyphens; they are not SQL comments.
    let is_comment = !stripped.starts_with("-----BEGIN ")
        && ["#", "//", "/*", "*", "<!--", "--"]
            .iter()
            .any(|prefix| stripped.starts_with(prefix));
    let scan_target = if is_comment {
        engine.comment_strip.replace(stripped, "").into_owned()
    } else {
        stripped.to_owned()
    };
    let comment_flag = is_comment.then_some(true);
    let snippet: String = stripped.chars().take(200).collect();
    let mut findings = Vec::new();

    for (signatures, severity, confidence) in [
        (&engine.critical, "critical", 0.99),
        (&engine.high, "high", 0.95),
    ] {
        for signature in signatures {
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

    let extension = effective_extension(file);
    if LOW_NOISE_EXTENSIONS.contains(&extension.as_str()) {
        return findings;
    }
    let scan_text = if HIGH_RISK_EXTENSIONS.contains(&extension.as_str()) {
        scan_target.clone()
    } else {
        engine.url.replace_all(&scan_target, " ").into_owned()
    };

    for match_ in engine.token.find_iter(&scan_text) {
        let token = match_.as_str();
        if looks_like_secret(token, 4.0) {
            findings.push(RawFinding {
                kind: "ENTROPY_CANDIDATE".to_owned(),
                line: lineno,
                snippet: snippet.clone(),
                token: token.to_owned(),
                token_start: None,
                token_end: None,
                evidence: Evidence::Entropy,
                severity: "info".to_owned(),
                confidence: entropy_confidence(shannon_entropy(token)),
                entropy: shannon_entropy(token),
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

    if HIGH_RISK_EXTENSIONS.contains(&extension.as_str()) {
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

fn looks_like_secret(value: &str, threshold: f64) -> bool {
    if value.len() < 20 {
        return false;
    }
    let unique: HashSet<u8> = value.bytes().collect();
    if unique.len() <= 3 {
        return false;
    }
    let has_upper = value.bytes().any(|byte| byte.is_ascii_uppercase());
    let has_lower = value.bytes().any(|byte| byte.is_ascii_lowercase());
    let has_digit = value.bytes().any(|byte| byte.is_ascii_digit());
    let has_special = value.bytes().any(|byte| !byte.is_ascii_alphanumeric());
    if [has_upper, has_lower, has_digit, has_special]
        .iter()
        .filter(|present| **present)
        .count()
        < 2
    {
        return false;
    }
    if !has_digit && !has_special {
        return false;
    }
    if has_lower && !has_upper && !has_digit {
        return false;
    }
    shannon_entropy(value) >= threshold
}

fn shannon_entropy(value: &str) -> f64 {
    if value.is_empty() {
        return 0.0;
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
