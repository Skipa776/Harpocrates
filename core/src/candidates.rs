//! Candidates for the ML step, and the names and context the Python classifier is given for them.
//!
//! Each rule here is the Python detector's (`cli/Harpocrates/core/detector.py` and
//! `classification.extract_var_name`), with the same pattern text, so both engines produce the same candidates.
//! `patterns()` lists every pattern by its Python name; a test compares the two lists.
//!
//! Python writes a few rules with a one-character lookbehind or lookahead, which the regex crate does not support.
//! [`Bounded`] checks those characters around each match instead, and `userinfo_key` scans for its `@` by hand.

use regex::Regex;

// Phase 2b and 2c: assignments to credential-named variables (quoted anywhere; unquoted in .env-style files).
pub(crate) const SENSITIVE_ASSIGNMENT: &str = r#"(?i)(?<![a-zA-Z])(?:pass(?:word|wd|w)?|pwd|usr(?:name)?|user|host|conn(?:ection|str)?|secret|token|key|auth|cred)[a-z0-9_]{0,40}\s*[:=]\s*['\"]([^'\"]{3,100})['\"]"#;
pub(crate) const ENV_ASSIGNMENT: &str = r#"(?i)(?<![a-zA-Z])(?:pass(?:word|wd|w)?|pwd|conn(?:ection|str)?|secret|token|key|auth|cred|url|endpoint|callback)[a-z0-9_]{0,40}\s*=\s*([^\s'\"]{3,200})"#;
// Phase 2d: credential shapes the entropy tokenizer cannot see.
const CONN_URI: &str =
    r#"(?<![a-z0-9+.\-])[a-z][a-z0-9+.\-]{0,19}://[^\s:/@'\"]*:[^\s@'\"/]{3,}@[^\s'\"]+"#;
const KV_PASSWORD: &str = r#"(?i)(?:^|[;&?\s'\"])(?:password|pwd|passwd)\s*=\s*([^;&'\"\s]{3,})"#;
const COLON_CREDENTIAL: &str = r##"(?i)(?<![a-z])(?:pass(?:word|wd)?|pwd|secret|token|api_?key|auth)[a-z0-9_]{0,40}['\"]?\s*:\s+([^\s'\"#]{6,200})\s*$"##;
const URL_TOKEN: &str = r##"(?i)[?&](?:token|access_token|api_key|apikey|key|sig|signature|auth|secret|client_secret)=([^&\s'\"#]{8,})"##;
const HEX_SECRET: &str = r"(?<![0-9a-f])[0-9a-f]{32,64}(?![0-9a-f])";
const QUOTED_LITERAL: &str = r#"['\"]([^'\"\s/]{8,64})['\"]"#;
const BEARER: &str = r#"(?i)\bbearer\s+([^\s'\"\\<>`]{6,200})"#;
const HEADER_SECRET: &str =
    r#"(?i)\b[\w-]{0,40}(?:key|token|sig|signature|secret)[\w-]{0,20}:\s*([^\s'\"\\`]{12,200})"#;
const URL_IN_LINE: &str = r#"(?i)(?<![a-z0-9+.\-])[a-z][a-z0-9+.\-]{0,19}://[^\s'\"<>`]{1,2000}"#;
const URL_PIECE: &str = r"[^/?&#]+";
const USERINFO_KEY: &str = r"[A-Za-z0-9_+\-]{16,200}(?=@)";
const WORDS_JOINED: &str = r"[A-Za-z]+(?:[-_.~][A-Za-z0-9]+)*";
const UUID: &str = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}";
const ASSIGNED_VALUE: &str = r#"[=:]\s*[\"']?([^\s'\"`]{12,120})"#;
const LETTER_RUN: &str = r"[A-Za-z]{5,}";
const PASSWORD_SYMBOLS: &str = "!@#$%^&*+=";
const SOUP_SYMBOLS: &str = "!@#$%^&*()[]{}<>;,?|~";
const URL_TRAILING: &str = ").,;:!?]}";
/// Characters left of a token that `extract_var_name` searches for `name =` (classification._VAR_NAME_WINDOW).
const VAR_NAME_WINDOW: usize = 256;
/// Characters on each side of a token that the classifier reads (classification._SEGMENT_CAP).
const CONTEXT_CHARS: usize = 512;

/// Every pattern and constant here, by its name in the Python detector.
pub(crate) fn patterns() -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = [
        ("_SENSITIVE_ASSIGNMENT_RE", SENSITIVE_ASSIGNMENT),
        ("_ENV_ASSIGNMENT_RE", ENV_ASSIGNMENT),
        ("_CONN_URI_RE", CONN_URI),
        ("_KV_PASSWORD_RE", KV_PASSWORD),
        ("_COLON_CREDENTIAL_RE", COLON_CREDENTIAL),
        ("_URL_TOKEN_RE", URL_TOKEN),
        ("_HEX_SECRET_RE", HEX_SECRET),
        ("_QUOTED_LITERAL_RE", QUOTED_LITERAL),
        ("_BEARER_RE", BEARER),
        ("_HEADER_SECRET_RE", HEADER_SECRET),
        ("_URL_IN_LINE_RE", URL_IN_LINE),
        ("_URL_PIECE_RE", URL_PIECE),
        ("_USERINFO_KEY_RE", USERINFO_KEY),
        ("_WORDS_JOINED_RE", WORDS_JOINED),
        ("_UUID_RE", UUID),
        ("_ASSIGNED_VALUE_RE", ASSIGNED_VALUE),
        ("_LETTER_RUN_RE", LETTER_RUN),
        ("_PASSWORD_SYMBOLS", PASSWORD_SYMBOLS),
        ("_SOUP_SYMBOLS", SOUP_SYMBOLS),
        ("_URL_TRAILING", URL_TRAILING),
    ]
    .into_iter()
    .map(|(name, pattern)| (name, pattern.to_owned()))
    .collect();
    out.push(("_VAR_NAME_WINDOW", VAR_NAME_WINDOW.to_string()));
    out.push(("_SEGMENT_CAP", CONTEXT_CHARS.to_string()));
    out
}

/// A Python pattern of the form `(?<![C])BODY` or `(?<![C])BODY(?![D])`, with optional leading flags.
///
/// A match of BODY counts when the character before it is not in C and the one after it is not in D. Checking the
/// lookahead after the greedy match equals Python's backtracking only when no shorter match could pass it; that
/// holds for the two patterns that use one (TELEGRAM_BOT_TOKEN has a fixed shape, and HEX_SECRET matches a whole
/// run of hex digits), and the parity tests cover both.
pub(crate) struct Bounded {
    behind: Option<Regex>,
    body: Regex,
    ahead: Option<Regex>,
}

impl Bounded {
    pub(crate) fn new(python: &str) -> Result<Self, regex::Error> {
        let flags = if python.starts_with("(?i)") {
            "(?i)"
        } else {
            ""
        };
        let mut body = &python[flags.len()..];
        let class = |text: &str| Regex::new(&format!("{flags}^{text}$"));
        let mut behind = None;
        if let Some(rest) = body.strip_prefix("(?<!") {
            let end = rest.find("])").map_or(0, |index| index + 1);
            behind = Some(class(&rest[..end])?);
            body = &rest[end + 1..];
        }
        let mut ahead = None;
        if let (true, Some(start)) = (body.ends_with("])"), body.rfind("(?![")) {
            ahead = Some(class(&body[start + 3..body.len() - 1])?);
            body = &body[..start];
        }
        Ok(Self {
            behind,
            body: Regex::new(&format!("{flags}{body}"))?,
            ahead,
        })
    }

    /// The pattern without its lookarounds: it matches wherever the Python pattern does, so it can prefilter.
    pub(crate) fn body(&self) -> &Regex {
        &self.body
    }

    /// Byte spans of capture group `group` for each match, like Python's `finditer`.
    pub(crate) fn spans(&self, text: &str, group: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut at = 0;
        while at <= text.len() {
            let Some(captures) = self.body.captures_at(text, at) else {
                break;
            };
            let whole = captures.get(0).expect("group 0 always matches");
            if self.allows(text, whole.start(), whole.end()) {
                if let Some(span) = captures.get(group) {
                    out.push((span.start(), span.end()));
                }
                at = whole.end().max(next_char(text, whole.start()));
            } else {
                at = next_char(text, whole.start());
            }
        }
        out
    }

    fn allows(&self, text: &str, start: usize, end: usize) -> bool {
        let outside = |check: &Option<Regex>, character: Option<char>| match (check, character) {
            (Some(class), Some(character)) => !class.is_match(character.encode_utf8(&mut [0; 4])),
            _ => true,
        };
        outside(&self.behind, text[..start].chars().next_back())
            && outside(&self.ahead, text[end..].chars().next())
    }
}

/// The byte index just after the character at `index`, or past the end.
fn next_char(text: &str, index: usize) -> usize {
    index + text[index..].chars().next().map_or(1, char::len_utf8)
}

/// Converts byte offsets of one text to character offsets, walking forward from the last answer.
pub(crate) struct CharIndex<'a> {
    text: &'a str,
    ascii: bool,
    byte: usize,
    chars: usize,
}

impl<'a> CharIndex<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        Self {
            text,
            ascii: text.is_ascii(),
            byte: 0,
            chars: 0,
        }
    }

    pub(crate) fn at(&mut self, byte: usize) -> usize {
        if self.ascii {
            return byte;
        }
        if byte < self.byte {
            (self.byte, self.chars) = (0, 0);
        }
        self.chars += self.text[self.byte..byte].chars().count();
        self.byte = byte;
        self.chars
    }
}

pub(crate) struct Candidates {
    pub(crate) sensitive_assignment: Bounded,
    pub(crate) env_assignment: Bounded,
    conn_uri: Bounded,
    kv_password: Regex,
    colon_credential: Bounded,
    url_token: Regex,
    hex_secret: Bounded,
    quoted_literal: Regex,
    bearer: Regex,
    header_secret: Regex,
    url_in_line: Bounded,
    url_piece: Regex,
    words_joined: Regex,
    uuid: Regex,
    assigned_value: Regex,
    letter_run: Regex,
    var_name_tail: Regex,
    identifier: Regex,
}

impl Candidates {
    pub(crate) fn compile() -> Result<Self, regex::Error> {
        Ok(Self {
            sensitive_assignment: Bounded::new(SENSITIVE_ASSIGNMENT)?,
            env_assignment: Bounded::new(ENV_ASSIGNMENT)?,
            conn_uri: Bounded::new(CONN_URI)?,
            kv_password: Regex::new(KV_PASSWORD)?,
            colon_credential: Bounded::new(COLON_CREDENTIAL)?,
            url_token: Regex::new(URL_TOKEN)?,
            hex_secret: Bounded::new(HEX_SECRET)?,
            quoted_literal: Regex::new(QUOTED_LITERAL)?,
            bearer: Regex::new(BEARER)?,
            header_secret: Regex::new(HEADER_SECRET)?,
            url_in_line: Bounded::new(URL_IN_LINE)?,
            url_piece: Regex::new(URL_PIECE)?,
            words_joined: Regex::new(&format!("^(?:{WORDS_JOINED})$"))?,
            uuid: Regex::new(&format!("^(?:{UUID})$"))?,
            assigned_value: Regex::new(ASSIGNED_VALUE)?,
            letter_run: Regex::new(LETTER_RUN)?,
            var_name_tail: Regex::new(r#"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*['\"]?$"#)?,
            identifier: Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$")?,
        })
    }

    /// Phase 2d (`_structured_candidates`): value and character offsets in `target`, in the Python order.
    pub(crate) fn structured(&self, target: &str) -> Vec<(String, usize, usize)> {
        let groups = |pattern: &Regex, group: usize| -> Vec<(usize, usize)> {
            pattern
                .captures_iter(target)
                .filter_map(|captures| captures.get(group).map(|span| (span.start(), span.end())))
                .collect()
        };
        let has_url = target.contains("://");
        let mut spans = Vec::new();
        if has_url {
            spans.extend(self.conn_uri.spans(target, 0));
        }
        spans.extend(groups(&self.kv_password, 1));
        spans.extend(groups(&self.url_token, 1));
        spans.extend(self.hex_secret.spans(target, 0));
        let shaped = [
            groups(&self.quoted_literal, 1),
            self.colon_credential.spans(target, 1),
            groups(&self.bearer, 1),
            groups(&self.header_secret, 1),
        ];
        spans.extend(
            shaped
                .into_iter()
                .flatten()
                .filter(|(start, end)| password_shaped(&target[*start..*end])),
        );
        spans.extend(
            groups(&self.assigned_value, 1)
                .into_iter()
                .filter(|(start, end)| self.symbol_soup(&target[*start..*end])),
        );
        if has_url {
            self.url_candidates(target, &mut spans);
        }
        let mut index = CharIndex::new(target);
        spans
            .into_iter()
            .map(|(start, end)| {
                (
                    target[start..end].to_owned(),
                    index.at(start),
                    index.at(end),
                )
            })
            .collect()
    }

    /// `_url_candidates`: a key-only userinfo (a Sentry DSN), and random-looking path pieces and query values.
    fn url_candidates(&self, target: &str, out: &mut Vec<(usize, usize)>) {
        for (url_start, url_end) in self.url_in_line.spans(target, 0) {
            let url = &target[url_start..url_end];
            let rest = url_start + url.find("://").expect("the pattern requires ://") + 3;
            let end = url_start + url.trim_end_matches(|c| URL_TRAILING.contains(c)).len();
            let run = target[rest..end]
                .bytes()
                .take_while(|byte| byte.is_ascii_alphanumeric() || b"_+-".contains(byte))
                .count();
            if (16..=200).contains(&run)
                && rest + run < end
                && target.as_bytes()[rest + run] == b'@'
            {
                out.push((rest, rest + run));
            }
            let query = target[rest..end]
                .find('?')
                .map_or(end, |index| rest + index);
            for piece in self.url_piece.find_iter(&target[rest..end]) {
                let mut start = rest + piece.start();
                if start == rest {
                    continue; // the host
                }
                let mut value = piece.as_str();
                if start > query
                    && let Some(cut) = value.find('=')
                {
                    // name=value only in the query; a path keeps its base64 padding
                    value = &value[cut + 1..];
                    start += cut + 1;
                }
                if self.random_url_piece(value) {
                    out.push((start, start + value.len()));
                }
            }
        }
    }

    fn random_url_piece(&self, value: &str) -> bool {
        value.chars().count() >= 16
            && password_shaped(value)
            && !self.words_joined.is_match(value)
            && !self.uuid.is_match(value)
    }

    /// Punctuation-heavy random values; code and prose have long letter runs that random strings lack.
    fn symbol_soup(&self, value: &str) -> bool {
        SOUP_SYMBOLS
            .chars()
            .filter(|symbol| value.contains(*symbol))
            .count()
            >= 3
            && value.chars().filter(|c| c.is_alphabetic()).count() >= 3
            && value.chars().any(is_digit)
            && !self.letter_run.is_match(value)
            && !value.contains("://")
    }

    /// `classification.extract_var_name`: the identifier assigned just left of the token's first occurrence.
    pub(crate) fn extract_var_name(&self, line: &str, token: &str) -> Option<String> {
        let index = line.find(token)?;
        if index == 0 {
            // A dotenv or shell KEY=VALUE token carries its key.
            let sep = [token.find('='), token.find(':')]
                .into_iter()
                .flatten()
                .filter(|position| *position > 0)
                .min()?;
            let candidate = token[..sep].trim();
            return self
                .identifier
                .is_match(candidate)
                .then(|| candidate.to_owned());
        }
        let start = back_chars(line, index, VAR_NAME_WINDOW);
        self.var_name_tail
            .captures(&line[start..index])
            .and_then(|captures| captures.get(1))
            .map(|name| name.as_str().to_owned())
    }
}

/// `_assigned_name`: everything left of an assignment's value, cut at the first `=` or `:`, trimmed.
pub(crate) fn assigned_name(text: &str, value_start: usize) -> Option<String> {
    let name = text[..value_start]
        .split('=')
        .next()
        .and_then(|left| left.split(':').next())
        .unwrap_or_default()
        .trim();
    (!name.is_empty()).then(|| name.to_owned())
}

/// The text around the token's first occurrence that the classifier reads (`_classify_by_line`), or None when the
/// token is not in the line.
pub(crate) fn context_window(line: &str, token: &str) -> Option<String> {
    let start = line.find(token)?;
    let end = start + token.len();
    let after = line[end..]
        .char_indices()
        .nth(CONTEXT_CHARS)
        .map_or(line.len(), |(index, _)| end + index);
    Some(line[back_chars(line, start, CONTEXT_CHARS)..after].to_owned())
}

/// The byte index `count` characters before `index`, or 0.
fn back_chars(text: &str, index: usize, count: usize) -> usize {
    text[..index]
        .char_indices()
        .rev()
        .nth(count - 1)
        .map_or(0, |(position, _)| position)
}

/// Letters and digits with an uppercase letter, a password symbol, or at least three letters and two digits.
fn password_shaped(value: &str) -> bool {
    let letters = value.chars().filter(|c| c.is_alphabetic()).count();
    let digits = value.chars().filter(|c| is_digit(*c)).count();
    if letters == 0 || digits == 0 {
        return false;
    }
    value.chars().any(char::is_uppercase)
        || value.chars().any(|c| PASSWORD_SYMBOLS.contains(c))
        || (letters >= 3 && digits >= 2)
}

/// Python's `str.isdigit` for ASCII; the nearest Unicode test otherwise.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || (!c.is_ascii() && c.is_numeric())
}
