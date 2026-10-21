"""
Context extraction for ML-based secrets verification.

Extracts surrounding code context to help distinguish between
true secrets and false positives like Git SHAs or UUIDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Pattern, Sequence, Tuple

from Harpocrates.core.classification import _VAR_NAME_WINDOW
from Harpocrates.ml.tokens import TokenMatch

if TYPE_CHECKING:
    from Harpocrates.core.result import Finding

# Variable name extraction patterns for different languages/formats
VAR_ASSIGNMENT_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    # Python/Ruby: var = "value" or var = 'value'
    ("python_assign", re.compile(r"(\w+)\s*=\s*[\"']")),
    # JavaScript/TypeScript: const/let/var name = "value"
    ("js_assign", re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*[\"']")),
    # Go: name := "value"
    ("go_assign", re.compile(r"(\w+)\s*:=\s*[\"']")),
    # Java/C#: String name = "value"
    ("java_assign", re.compile(r"(?:String|string)\s+(\w+)\s*=\s*[\"']")),
    # JSON/Dict key: "key": "value" or 'key': 'value'
    ("json_key", re.compile(r"[\"'](\w+)[\"']\s*:\s*[\"']")),
    # YAML: key: value or key: "value"
    ("yaml_key", re.compile(r"^(\w+)\s*:\s*[\"']?")),
    # Shell export: export VAR=value or export VAR="value"
    ("shell_export", re.compile(r"export\s+(\w+)\s*=")),
    # .env file: VAR=value
    ("env_assign", re.compile(r"^(\w+)\s*=")),
    # Function argument: func(api_key="value")
    ("func_arg", re.compile(r"(\w+)\s*=\s*[\"']")),
]

# Patterns indicating safe (non-secret) context
SAFE_CONTEXT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\b(test|mock|fake|example|sample|dummy|placeholder)\b", re.I),
    re.compile(r"\b(commit|sha|hash|digest|checksum|rev|revision)\b", re.I),
    re.compile(r"\b(uuid|guid|id|identifier)\b", re.I),
    re.compile(r"\bgit\s+(log|show|rev-parse|commit|diff)\b", re.I),
    re.compile(r"\b(md5|sha1|sha256|sha512|blake2)\b", re.I),
]

# Patterns indicating risky (likely secret) context
RISKY_CONTEXT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\b(secret|password|credential|private|auth)\b", re.I),
    re.compile(r"\b(api[_-]?key|access[_-]?key|token)\b", re.I),
    re.compile(r"\b(bearer|authorization)\b", re.I),
    re.compile(r"\b(AWS|AZURE|GCP|STRIPE|OPENAI|GITHUB)\b"),
]

# File extensions with higher risk scores
HIGH_RISK_EXTENSIONS = {".env", ".pem", ".key", ".secret", ".credentials"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"}

# File extensions where entropy is pure noise — structured markup/style/vector files
# generate webpack fingerprints, CSS class hashes, and SVG path data that satisfy
# the entropy threshold but are never credentials. Regex tier still runs so
# accidentally hardcoded API keys in <script> blocks are still caught.
LOW_NOISE_EXTENSIONS = frozenset({".html", ".htm", ".css", ".scss", ".sass", ".svg", ".xml"})
TEST_PATH_PATTERNS = [
    re.compile(r"(^|[/\\])tests?[/\\]", re.I),
    re.compile(r"(^|[/\\])__tests__[/\\]", re.I),
    re.compile(r"(^|[/\\])spec[/\\]", re.I),
    re.compile(r"_test\.py$", re.I),
    re.compile(r"\.test\.[jt]sx?$", re.I),
    re.compile(r"\.spec\.[jt]sx?$", re.I),
    re.compile(r"test_\w+\.py$", re.I),
]


@dataclass
class CodeContext:
    """
    Description: Hold the surrounding code and metadata that help the ML model tell true secrets from false positives.
    Parameters:
        line_content: Line containing the candidate.
        lines_before: Lines preceding the candidate.
        lines_after: Lines following the candidate.
        file_path: Optional source file path.
        file_type: Optional language identifier.
        in_test_file: Whether the path identifies a test file.
        in_comment: Whether the current line is a comment.
        in_string_literal: Whether the candidate is in a string literal.
        line_number: One-based finding line number.
        total_lines: Total source line count.
        token_match: Optional token text and span metadata.
    Output: Instances provide context lines, file metadata, and derived file-risk properties.
    """

    line_content: str
    lines_before: List[str] = field(default_factory=list)
    lines_after: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    in_test_file: bool = False
    in_comment: bool = False
    in_string_literal: bool = True  # Most secrets are in strings
    line_number: Optional[int] = None  # 1-based line number in file
    total_lines: Optional[int] = None  # Total lines in file
    token_match: Optional[TokenMatch] = None

    @property
    def file_extension(self) -> Optional[str]:
        """
        Description: Extract the lowercase path suffix for file-type checks.
        Parameters: None
        Output: File suffix, or None when no path is available.
        Get file extension from path.
        """
        if self.file_path:
            return Path(self.file_path).suffix.lower()
        return None

    @property
    def file_name(self) -> Optional[str]:
        """
        Description: Extract the final path component for filename checks.
        Parameters: None
        Output: File name, or None when no path is available.
        Get file name from path.
        """
        if self.file_path:
            return Path(self.file_path).name
        return None

    @property
    def is_config_file(self) -> bool:
        """
        Description: Recognize configuration files using extensions and filename hints.
        Parameters: None
        Output: True for matching configuration files, otherwise False.
        Check if file is a configuration file.
        """
        ext = self.file_extension
        name = self.file_name
        if ext in CONFIG_EXTENSIONS:
            return True
        if ext == ".env" or (name and name.startswith(".env")):
            return True
        if name and any(x in name.lower() for x in ["config", "settings", "credential"]):
            return True
        return False

    @property
    def is_high_risk_file(self) -> bool:
        """
        Description: Check the extension against high-risk secret file types.
        Parameters: None
        Output: True for a high-risk extension, otherwise False.
        Check if file type indicates high risk for secrets.
        """
        ext = self.file_extension
        return ext in HIGH_RISK_EXTENSIONS

    @property
    def full_context(self) -> str:
        """
        Description: Join the surrounding lines to supply text for context features.
        Parameters: None
        Output: Before, current, and after lines joined by newline characters.
        Get all context lines as a single string.
        """
        all_lines = self.lines_before + [self.line_content] + self.lines_after
        return "\n".join(all_lines)


def feature_var_name(line: str, token: str) -> Optional[str]:
    """
    Description: Search the assignment patterns just before a token to find its variable name.
    Parameters:
        line: The line of code
        token: The detected token
    Output: Matched variable name, or None if the token or assignment is absent.
    """
    # Find position of token in line
    token_pos = line.find(token)
    if token_pos == -1:
        return None

    # Only the text just left of the token can hold `name =`. Searching the whole prefix is
    # quadratic in word-character runs (minified JS, inline base64). `search(line, start, end)`
    # bounds the scan while `^` still anchors at the real line start.
    window_start = max(0, token_pos - _VAR_NAME_WINDOW)

    for pattern_name, pattern in VAR_ASSIGNMENT_PATTERNS:
        match = pattern.search(line, window_start, token_pos)
        if match:
            return match.group(1)

    return None


def is_comment_line(line: str) -> bool:
    """
    Description: Check stripped text for supported comment prefixes to label context.
    Parameters:
        line: Line to inspect.
    Output: True when a supported comment prefix matches, otherwise False.
    """
    stripped = line.strip()
    comment_prefixes = ("#", "//", "/*", "*", "<!--", "--", ";", "rem ")
    return any(stripped.startswith(prefix) for prefix in comment_prefixes)


def detect_file_type(file_path: Optional[str]) -> Optional[str]:
    """
    Description: Map a file extension to a language identifier for context metadata.
    Parameters:
        file_path: Path to the file
    Output: Language identifier, or None for a missing path or unknown extension.
    """
    if not file_path:
        return None

    ext = Path(file_path).suffix.lower()
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".ini": "ini",
        ".env": "dotenv",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
    }
    return extension_map.get(ext)


def is_test_file(file_path: Optional[str]) -> bool:
    """
    Description: Match test-path patterns to identify test context.
    Parameters:
        file_path: Optional source file path.
    Output: True for a matching test path, otherwise False, including missing paths.
    """
    if not file_path:
        return False
    return any(pattern.search(file_path) for pattern in TEST_PATH_PATTERNS)


def extract_context(
    content: str,
    line_number: int,
    file_path: Optional[str] = None,
    context_lines: int = 3,
) -> CodeContext:
    """
    Description: Split content and extract the neighboring lines that describe a finding.
    Parameters:
        content: Full file content or text blob
        line_number: 1-based line number of the finding
        file_path: Optional file path for metadata extraction
        context_lines: Number of lines before/after to include
    Output: CodeContext containing surrounding code and file metadata.
    """
    return extract_context_from_lines(
        content.splitlines(),
        line_number,
        file_path,
        context_lines=context_lines,
    )


def extract_context_from_lines(
    lines: Sequence[str],
    line_number: int,
    file_path: Optional[str] = None,
    context_lines: int = 3,
) -> CodeContext:
    """
    Description: Extract context from lines already split by a batch caller, so batches do not re-split the same
        content.
    Parameters:
        lines: Previously split source lines.
        line_number: One-based finding line number.
        file_path: Optional source file path.
        context_lines: Number of neighboring lines on each side.
    Output: CodeContext; an out-of-bounds line produces empty line content and file metadata.
    """
    total_lines = len(lines)

    # Adjust to 0-based indexing
    line_idx = line_number - 1

    # Handle out of bounds
    if line_idx < 0 or line_idx >= len(lines):
        return CodeContext(
            line_content="",
            file_path=file_path,
            file_type=detect_file_type(file_path),
            in_test_file=is_test_file(file_path),
            line_number=line_number,
            total_lines=total_lines,
        )

    line_content = lines[line_idx]

    # Get context lines
    start_idx = max(0, line_idx - context_lines)
    end_idx = min(len(lines), line_idx + context_lines + 1)

    lines_before = list(lines[start_idx:line_idx])
    lines_after = list(lines[line_idx + 1 : end_idx])

    return CodeContext(
        line_content=line_content,
        lines_before=lines_before,
        lines_after=lines_after,
        file_path=file_path,
        file_type=detect_file_type(file_path),
        in_test_file=is_test_file(file_path),
        in_comment=is_comment_line(line_content),
        line_number=line_number,
        total_lines=total_lines,
    )


def extract_context_from_finding(
    finding: Finding,
    full_content: Optional[str] = None,
    context_lines: int = 3,
) -> CodeContext:
    """
    Description: Build context from a finding, reading its file when full content is unavailable.
    Parameters:
        finding: Finding object with file/line info
        full_content: Optional full file content (reads from disk if not provided)
        context_lines: Number of lines before/after to include
    Output: CodeContext; may read a UTF-8 file, ignoring decoding errors. File I/O errors fall back to the snippet.
    """
    if finding.line is None:
        # No line info, create minimal context
        return CodeContext(
            line_content=finding.snippet,
            file_path=finding.file,
            file_type=detect_file_type(finding.file),
            in_test_file=is_test_file(finding.file),
        )

    # Read file content if not provided
    if full_content is None and finding.file:
        try:
            full_content = Path(finding.file).read_text(encoding="utf-8", errors="ignore")
        except (OSError, IOError):
            full_content = finding.snippet

    if full_content is None:
        full_content = finding.snippet

    return extract_context(
        content=full_content,
        line_number=finding.line,
        file_path=finding.file,
        context_lines=context_lines,
    )


__all__ = [
    "CodeContext",
    "extract_context",
    "extract_context_from_lines",
    "extract_context_from_finding",
    "feature_var_name",
    "is_comment_line",
    "is_test_file",
    "detect_file_type",
    "SAFE_CONTEXT_PATTERNS",
    "RISKY_CONTEXT_PATTERNS",
    "VAR_ASSIGNMENT_PATTERNS",
]
