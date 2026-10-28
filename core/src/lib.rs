//! Native regex and entropy scanner for Harpocrates.
//!
//! The crate owns candidate generation, every regex and entropy rule, with the Python detector's patterns and
//! order (`patterns()` lists them; a Python test compares the two). Python owns category inference, context
//! features and ML verification.

mod candidates;

use candidates::{CharIndex, assigned_name, context_window};
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::thread;

/// 2: `memory_cap` replaced the read limits `max_bytes` and `max_file_size` (FR-CORE-05).
/// 3: candidates carry the Python detector's variable name and a `context` window, and the Phase 2d candidates exist.
pub const PROTOCOL_VERSION: u32 = 3;
/// The most bytes one file's scan holds (FR-CORE-05); every file is scanned whole, in pieces.
pub const MEMORY_CAP: usize = 64 * 1024 * 1024;
/// Bytes that two pieces of a long line share: longer than any token the regex and entropy tiers match.
pub const PIECE_OVERLAP: usize = 64 * 1024;

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
    /// The line around the token's first occurrence, for the classifier; None for regex signatures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ScanOptions {
    pub file: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileScanRequest {
    pub path: PathBuf,
    pub memory_cap: Option<usize>,
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
    pub memory_cap: usize,
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

const CRITICAL_SIGNATURES: &[(&str, &str)] = &[
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
        r"\bxox[pboar]-[0-9]{10,13}-[a-zA-Z0-9\-]{24,48}\b",
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

const HIGH_SIGNATURES: &[(&str, &str)] = &[
    (
        "TELEGRAM_BOT_TOKEN",
        r"(?<![0-9])[0-9]{8,10}:[A-Za-z0-9_\-]{35}(?![A-Za-z0-9_\-])",
    ),
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
    // Matched by Matcher::OpenAiLegacy, which is written from this pattern: change both together.
    (
        "OPENAI_API_KEY_LEGACY",
        r"\bsk-(?!proj-)(?!ant-api03-)(?![a-zA-Z0-9]{48}\b)[A-Za-z0-9_-]{16,}\b",
    ),
];

const TOKEN: &str = r"[A-Za-z0-9+/=_\-]{20,}";
const URL: &str = r"(?:https?://|data:)\S+";
const PEM_BODY: &str = r"^[A-Za-z0-9+/]{64}$";
const COMMENT_STRIP: &str = r"^(?:#+|//+|/\*+|\*+|<!--|--)\s*";
const PROSE_FILTER: &str = r#"[=:'\"]"#;

/// Every pattern and constant the scan uses, by its Python name, in the Python order.
pub fn patterns() -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = CRITICAL_SIGNATURES
        .iter()
        .chain(HIGH_SIGNATURES)
        .chain(&[
            ("_TOKEN_RE", TOKEN),
            ("_URL_RE", URL),
            ("_PEM_BODY_RE", PEM_BODY),
            ("_COMMENT_STRIP_RE", COMMENT_STRIP),
            ("_PROSE_FILTER_RE", PROSE_FILTER),
        ])
        .map(|(name, pattern)| (*name, (*pattern).to_owned()))
        .collect();
    out.extend(candidates::patterns());
    out
}

/// How a signature finds its matches.
enum Matcher {
    Plain(Regex),
    /// A one-character lookbehind or lookahead around the match (TELEGRAM_BOT_TOKEN).
    Bounded(candidates::Bounded),
    /// `\bsk-`, three lookaheads, then the body.
    OpenAiLegacy {
        start: Regex,
        body: Regex,
        all_48: Regex,
    },
}

struct Signature {
    name: &'static str,
    matcher: Matcher,
}

impl Signature {
    /// The signature, and a pattern without lookarounds that matches wherever it does, for the prefilter set.
    fn compile(name: &'static str, python: &str) -> Result<(Self, String), regex::Error> {
        let (matcher, prefilter) = if name == "OPENAI_API_KEY_LEGACY" {
            let matcher = Matcher::OpenAiLegacy {
                start: Regex::new(r"\bsk-")?,
                body: Regex::new(r"[A-Za-z0-9_-]{16,}\b")?,
                all_48: Regex::new(r"[a-zA-Z0-9]{48}\b")?,
            };
            (matcher, r"\bsk-[A-Za-z0-9_-]{16,}\b".to_owned())
        } else if python.contains("(?<!") || python.contains("(?!") {
            let bounded = candidates::Bounded::new(python)?;
            let prefilter = bounded.body().as_str().to_owned();
            (Matcher::Bounded(bounded), prefilter)
        } else {
            (Matcher::Plain(Regex::new(python)?), python.to_owned())
        };
        Ok((Self { name, matcher }, prefilter))
    }

    /// Byte spans of the matches, like Python's `finditer`.
    fn spans(&self, text: &str) -> Vec<(usize, usize)> {
        match &self.matcher {
            Matcher::Plain(pattern) => pattern
                .find_iter(text)
                .map(|found| (found.start(), found.end()))
                .collect(),
            Matcher::Bounded(bounded) => bounded.spans(text, 0),
            Matcher::OpenAiLegacy {
                start,
                body,
                all_48,
            } => {
                // A match of `pattern` that starts exactly at `at`, with the text before it as context.
                let at_start = |pattern: &Regex, at: usize| {
                    pattern
                        .find_at(text, at)
                        .filter(|found| found.start() == at)
                };
                let mut out = Vec::new();
                let mut at = 0;
                while let Some(prefix) = start.find_at(text, at) {
                    let rest = &text[prefix.end()..];
                    let excluded = rest.starts_with("proj-")
                        || rest.starts_with("ant-api03-")
                        || at_start(all_48, prefix.end()).is_some();
                    match (!excluded).then(|| at_start(body, prefix.end())).flatten() {
                        Some(found) => {
                            out.push((prefix.start(), found.end()));
                            at = found.end();
                        }
                        None => at = prefix.start() + 1,
                    }
                }
                out
            }
        }
    }
}

fn compile_tier(
    definitions: &[(&'static str, &'static str)],
) -> Result<(Vec<Signature>, RegexSet), regex::Error> {
    let mut signatures = Vec::new();
    let mut prefilters = Vec::new();
    for (name, python) in definitions {
        let (signature, prefilter) = Signature::compile(name, python)?;
        signatures.push(signature);
        prefilters.push(prefilter);
    }
    Ok((signatures, RegexSet::new(prefilters)?))
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
    candidates: candidates::Candidates,
}

impl Engine {
    fn compile() -> Result<Self, regex::Error> {
        let (critical, critical_set) = compile_tier(CRITICAL_SIGNATURES)?;
        let (high, high_set) = compile_tier(HIGH_SIGNATURES)?;
        Ok(Self {
            critical,
            critical_set,
            high,
            high_set,
            token: Regex::new(TOKEN)?,
            url: Regex::new(URL)?,
            pem_body: Regex::new(PEM_BODY)?,
            comment_strip: Regex::new(COMMENT_STRIP)?,
            prose_filter: Regex::new(PROSE_FILTER)?,
            candidates: candidates::Candidates::compile()?,
        })
    }
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
    memory_cap: Option<usize>,
) -> Result<Vec<RawFinding>, ScanError> {
    scan_file_response(path.as_ref(), memory_cap).map(|response| response.findings)
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
            |file| match scan_file_response(&file.path, file.memory_cap) {
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

    let files = scan_directory_paths(&paths, request.memory_cap)?;

    Ok(BatchScanResponse {
        protocol_version: PROTOCOL_VERSION,
        files,
    })
}

fn scan_directory_paths(
    paths: &[PathBuf],
    memory_cap: usize,
) -> Result<Vec<FileScanResponse>, ScanError> {
    let workers = configured_worker_count(paths.len());
    if workers <= 1 {
        return Ok(paths
            .iter()
            .cloned()
            .map(|path| scan_path(path, memory_cap))
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
                        .map(|path| scan_path(path, memory_cap))
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
    // HARPO_RUST_WORKERS first, then the old HARPOCRATES_RUST_WORKERS (FR-NAME-04).
    let configured = std::env::var("HARPO_RUST_WORKERS")
        .or_else(|_| std::env::var("HARPOCRATES_RUST_WORKERS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(available);
    configured.min(available).min(8).min(task_count)
}

fn scan_path(path: PathBuf, memory_cap: usize) -> FileScanResponse {
    match scan_file_response(&path, Some(memory_cap)) {
        Ok(response) => response,
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
    memory_cap: Option<usize>,
) -> Result<FileScanResponse, ScanError> {
    let open = || {
        File::open(path)
            .map_err(|error| ScanError::new(format!("failed to read {}: {error}", path.display())))
    };
    let read_error =
        |error: io::Error| ScanError::new(format!("failed to read {}: {error}", path.display()));
    let mut sample = Vec::with_capacity(1024);
    open()?
        .take(1024)
        .read_to_end(&mut sample)
        .map_err(read_error)?;
    if looks_binary(&sample) {
        return Ok(FileScanResponse {
            path: path.to_path_buf(),
            findings: Vec::new(),
            scanned: true,
            line_count: 0,
            bytes_scanned: sample.len(),
            error: None,
        });
    }
    let file = path.to_string_lossy().into_owned();
    let reader = BufReader::with_capacity(PIECE_OVERLAP, open()?);
    let (findings, line_count, bytes_scanned) =
        scan_reader(reader, memory_cap.unwrap_or(MEMORY_CAP), Some(&file)).map_err(read_error)?;
    Ok(FileScanResponse {
        path: path.to_path_buf(),
        findings,
        scanned: true,
        line_count,
        bytes_scanned,
        error: None,
    })
}

/// One piece of a line: its 1-based line number, the character offset where it starts in that line, and its text.
struct Piece {
    line: usize,
    offset: usize,
    text: String,
}

/// A file's lines read lazily in pieces of at most `limit` bytes (FR-CORE-05). A longer line is cut on a character
/// boundary into pieces that share PIECE_OVERLAP bytes, so a scan of any size holds one piece at a time. Lines end at
/// `\n`, and a `\r` before it is dropped, as `str::lines` does; invalid UTF-8 becomes U+FFFD.
struct LinePieces<R: BufRead> {
    reader: R,
    limit: usize,
    line: usize,
    chars: usize,
    carry: String,
    spill: Vec<u8>,
    bytes: usize,
    done: bool,
}

impl<R: BufRead> LinePieces<R> {
    fn new(reader: R, limit: usize) -> Self {
        LinePieces {
            reader,
            limit,
            line: 1,
            chars: 0,
            carry: String::new(),
            spill: Vec::new(),
            bytes: 0,
            done: false,
        }
    }

    fn next_piece(&mut self) -> io::Result<Option<Piece>> {
        if self.done {
            return Ok(None);
        }
        let budget = self.limit.saturating_sub(self.carry.len()).max(4);
        let mut raw = std::mem::take(&mut self.spill);
        let (mut ended, mut eof) = (false, false);
        while raw.len() < budget {
            let available = self.reader.fill_buf()?;
            if available.is_empty() {
                eof = true;
                break;
            }
            let window = &available[..available.len().min(budget - raw.len())];
            let (take, newline) = match window.iter().position(|byte| *byte == b'\n') {
                Some(index) => (index + 1, true),
                None => (window.len(), false),
            };
            raw.extend_from_slice(&window[..take]);
            self.reader.consume(take);
            self.bytes += take;
            if newline {
                ended = true;
                break;
            }
        }
        if eof && raw.is_empty() {
            self.done = true;
            return Ok(None);
        }
        if ended {
            raw.pop();
            if raw.last() == Some(&b'\r') {
                raw.pop();
            }
        } else if !eof {
            let cut = utf8_cut(&raw);
            self.spill = raw.split_off(cut);
        }
        let fresh = String::from_utf8_lossy(&raw).into_owned();
        let carried = self.carry.chars().count();
        let mut text = std::mem::take(&mut self.carry);
        text.push_str(&fresh);
        let piece = Piece {
            line: self.line,
            offset: self.chars - carried,
            text,
        };
        if ended || eof {
            self.line += 1;
            self.chars = 0;
            self.done = eof;
        } else {
            self.chars += fresh.chars().count();
            let mut start = piece.text.len().saturating_sub(PIECE_OVERLAP);
            while !piece.text.is_char_boundary(start) {
                start += 1;
            }
            self.carry = piece.text[start..].to_owned();
        }
        Ok(Some(piece))
    }
}

/// Where to cut bytes read from the middle of a line so that no UTF-8 character is split: before a final character
/// whose bytes are not all there.
fn utf8_cut(raw: &[u8]) -> usize {
    let tail = raw.len().saturating_sub(3);
    for start in (tail..raw.len()).rev() {
        let byte = raw[start];
        if byte & 0b1100_0000 == 0b1000_0000 {
            continue;
        }
        let width = match byte {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf7 => 4,
            _ => 1,
        };
        return if start + width > raw.len() {
            start
        } else {
            raw.len()
        };
    }
    raw.len()
}

/// The most bytes of a line one piece holds, for a memory cap: a quarter of it, and at least twice PIECE_OVERLAP.
fn piece_bytes(memory_cap: usize) -> usize {
    (memory_cap / 4).max(2 * PIECE_OVERLAP)
}

/// Scan text read from `reader` line by line, in bounded pieces (FR-CORE-05). Returns the findings, with offsets
/// counted as in a scan of each whole line; the line count; and the bytes read. A finding in the overlap of two
/// pieces is kept once. On a line longer than one piece, the entropy tier runs per piece, so such a line can give
/// more candidates than a whole-line scan, never fewer regex findings.
fn scan_reader<R: BufRead>(
    reader: R,
    memory_cap: usize,
    file: Option<&str>,
) -> io::Result<(Vec<RawFinding>, usize, usize)> {
    let engine = engine().map_err(|error| io::Error::other(error.to_string()))?;
    let extension = effective_extension(file);
    let mut pieces = LinePieces::new(reader, piece_bytes(memory_cap));
    let mut findings = Vec::new();
    let (mut in_block_comment, mut was_in_block, mut is_comment) = (false, false, false);
    let (mut base, mut lines) = (0usize, 0usize);
    let mut earlier: HashSet<(String, String, Option<usize>)> = HashSet::new();
    while let Some(piece) = pieces.next_piece()? {
        let stripped = piece.text.trim();
        if piece.offset == 0 {
            lines += 1;
            earlier.clear();
            was_in_block = in_block_comment && !stripped.starts_with("/*");
            (base, is_comment) = line_base(engine, &piece.text);
        }
        let line_findings = if piece.offset == 0 {
            scan_line(engine, &piece.text, piece.line, &extension)
        } else {
            scan_prepared(
                engine,
                &piece.text,
                Cow::Borrowed(&piece.text),
                is_comment,
                piece.line,
                &extension,
            )
        };
        let mut found = HashSet::new();
        for mut finding in line_findings {
            if was_in_block && finding.in_comment.is_none() {
                finding.in_comment = Some(true);
            }
            if piece.offset > 0 {
                let shift = piece.offset - base.min(piece.offset);
                finding.token_start = finding.token_start.map(|start| start + shift);
                finding.token_end = finding.token_end.map(|end| end + shift);
            }
            // A regex finding has line offsets; a candidate's count from URL-stripped text, so it is keyed by token.
            let start = matches!(finding.evidence, Evidence::Regex)
                .then_some(finding.token_start)
                .flatten();
            let key = (finding.kind.clone(), finding.token.clone(), start);
            if !earlier.contains(&key) {
                findings.push(finding);
            }
            found.insert(key);
        }
        earlier.extend(found);
        let fresh = if piece.offset > 0 {
            let mut at = PIECE_OVERLAP.min(piece.text.len());
            while !piece.text.is_char_boundary(at) {
                at += 1;
            }
            &piece.text[at..]
        } else {
            stripped
        };
        let (opens, closes) = (fresh.matches("/*").count(), fresh.matches("*/").count());
        in_block_comment = (in_block_comment && closes == 0) || opens > closes;
    }
    Ok((findings, lines, pieces.bytes))
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

/// What the line scan matches against: the trimmed line without a comment prefix, and whether the line is a comment.
/// The target is a suffix of the trimmed line, so token offsets count from where it starts.
fn scan_target<'a>(engine: &Engine, line: &'a str) -> (&'a str, Cow<'a, str>, bool) {
    let stripped = line.trim();
    // PEM headers begin with five hyphens; they are not SQL comments.
    let is_comment = !stripped.starts_with("-----BEGIN ")
        && ["#", "//", "/*", "*", "<!--", "--"]
            .iter()
            .any(|prefix| stripped.starts_with(prefix));
    let target = if is_comment {
        engine.comment_strip.replace(stripped, "")
    } else {
        Cow::Borrowed(stripped)
    };
    (stripped, target, is_comment)
}

/// Where a line's token offsets count from, for its later pieces (FR-CORE-05): the characters dropped before the
/// scan target, and whether the line is a comment.
fn line_base(engine: &Engine, line: &str) -> (usize, bool) {
    let (stripped, target, is_comment) = scan_target(engine, line);
    let leading = line.chars().count() - line.trim_start().chars().count();
    (
        leading + stripped.chars().count() - target.chars().count(),
        is_comment,
    )
}

fn scan_line(engine: &Engine, line: &str, lineno: usize, extension: &str) -> Vec<RawFinding> {
    let (stripped, target, is_comment) = scan_target(engine, line);
    scan_prepared(engine, stripped, target, is_comment, lineno, extension)
}

/// The detection tiers over a prepared target, in the Python detector's order (`_scan_line`). A later piece of a long
/// line comes here as it is, with the line's comment flag, so its offsets count from the piece's start.
fn scan_prepared(
    engine: &Engine,
    stripped: &str,
    scan_target: Cow<'_, str>,
    is_comment: bool,
    lineno: usize,
    extension: &str,
) -> Vec<RawFinding> {
    if stripped.trim().is_empty() {
        return Vec::new();
    }
    let target: &str = &scan_target;
    let line = LineFacts {
        lineno,
        snippet: stripped.chars().take(200).collect(),
        in_comment: is_comment.then_some(true),
    };
    let mut findings = Vec::new();

    // Phase 1: CRITICAL, then HIGH signatures.
    let mut index = CharIndex::new(target);
    for (signatures, signature_set, severity, confidence) in [
        (&engine.critical, &engine.critical_set, "critical", 0.99),
        (&engine.high, &engine.high_set, "high", 0.95),
    ] {
        for signature_index in signature_set.matches(target).iter() {
            let signature = &signatures[signature_index];
            for (start, end) in signature.spans(target) {
                let token = &target[start..end];
                findings.push(RawFinding {
                    kind: signature.name.to_owned(),
                    line: lineno,
                    snippet: line.snippet.clone(),
                    token: token.to_owned(),
                    token_start: Some(index.at(start)),
                    token_end: Some(index.at(end)),
                    evidence: Evidence::Regex,
                    severity: severity.to_owned(),
                    confidence,
                    entropy: shannon_entropy(token),
                    in_comment: line.in_comment,
                    var_name: None,
                    context: None,
                });
            }
        }
    }

    // Phase 2 runs only when no signature matched.
    if !findings.is_empty()
        || (is_comment && !engine.prose_filter.is_match(target))
        || engine.pem_body.is_match(target)
        || LOW_NOISE_EXTENSIONS.contains(&extension)
    {
        return findings;
    }
    let high_risk = HIGH_RISK_EXTENSIONS.contains(&extension);
    // URLs are credentials in .env-style files, and noise elsewhere.
    let scan_text: Cow<'_, str> = if high_risk {
        Cow::Borrowed(target)
    } else {
        engine.url.replace_all(target, " ")
    };
    let candidates = &engine.candidates;

    // Phase 2a: high-entropy tokens.
    for found in engine.token.find_iter(&scan_text) {
        let token = found.as_str();
        if let Some(entropy) = secret_entropy(token, 4.0) {
            findings.push(RawFinding {
                kind: "ENTROPY_CANDIDATE".to_owned(),
                line: lineno,
                snippet: line.snippet.clone(),
                token: token.to_owned(),
                token_start: None,
                token_end: None,
                evidence: Evidence::Entropy,
                severity: "info".to_owned(),
                confidence: entropy_confidence(entropy),
                entropy,
                in_comment: line.in_comment,
                var_name: candidates.extract_var_name(target, token),
                context: context_window(target, token),
            });
        }
    }

    // Each later phase skips a value already found on this line.
    let mut found_tokens: HashSet<String> = findings
        .iter()
        .map(|finding| finding.token.clone())
        .collect();
    let mut text_index = CharIndex::new(&scan_text);
    // Phase 2b: quoted values of credential-named variables.
    for (start, end) in candidates.sensitive_assignment.spans(&scan_text, 1) {
        let value = &scan_text[start..end];
        if found_tokens.insert(value.to_owned()) {
            let span = (text_index.at(start), text_index.at(end));
            findings.push(line.candidate(
                "ML_CANDIDATE",
                Evidence::Ml,
                value,
                span,
                assigned_name(&scan_text, start),
                context_window(target, value),
            ));
        }
    }
    // Phase 2d: credential shapes the tokenizer cannot see.
    for (value, start, end) in candidates.structured(target) {
        if found_tokens.contains(&value) {
            continue;
        }
        let (var_name, context) = (
            candidates.extract_var_name(target, &value),
            context_window(target, &value),
        );
        findings.push(line.candidate(
            "ML_CANDIDATE",
            Evidence::Ml,
            &value,
            (start, end),
            var_name,
            context,
        ));
        found_tokens.insert(value);
    }
    // Phase 2c: unquoted KEY=VALUE in .env-style files.
    if high_risk {
        for (start, end) in candidates.env_assignment.spans(&scan_text, 1) {
            let value = &scan_text[start..end];
            if found_tokens.insert(value.to_owned()) {
                let span = (text_index.at(start), text_index.at(end));
                findings.push(line.candidate(
                    "ENV_ASSIGNMENT",
                    Evidence::Regex,
                    value,
                    span,
                    assigned_name(&scan_text, start),
                    context_window(target, value),
                ));
            }
        }
    }

    findings
}

/// What every finding on one line shares.
struct LineFacts {
    lineno: usize,
    snippet: String,
    in_comment: Option<bool>,
}

impl LineFacts {
    /// A Phase 2 candidate at character offsets `span`; Python sets its severity and confidence.
    fn candidate(
        &self,
        kind: &str,
        evidence: Evidence,
        token: &str,
        span: (usize, usize),
        var_name: Option<String>,
        context: Option<String>,
    ) -> RawFinding {
        RawFinding {
            kind: kind.to_owned(),
            line: self.lineno,
            snippet: self.snippet.clone(),
            token: token.to_owned(),
            token_start: Some(span.0),
            token_end: Some(span.1),
            evidence,
            severity: "info".to_owned(),
            confidence: 0.5,
            entropy: shannon_entropy(token),
            in_comment: self.in_comment,
            var_name,
            context,
        }
    }
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
