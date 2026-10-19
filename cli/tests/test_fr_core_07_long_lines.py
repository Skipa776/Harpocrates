"""FR-CORE-07: feature extraction runs in time linear in line length (minified JS, inline base64)."""

import re
import time

import pytest

from Harpocrates.core.result import EvidenceType, Finding, Severity
from Harpocrates.ml.context import extract_context, feature_var_name
from Harpocrates.ml.features import extract_features

_TOKEN = "tok_9fQ2mZ7xL"  # generated canary, not a real credential
_LINE_LEN = 200_000
_TIME_BOUND_SECONDS = 0.5

_B64_UNIT = "QUJDREVGR0hJSktMTU5PUA+/"


def _long_runs(kind: str) -> str:
    """
    Description: Build a long run of one character class to stress the extractors.
    Parameters:
        kind: "word" for one unbroken word-character run, "base64" for a base64-ish run.
    Output: a string of _LINE_LEN characters.
    """
    if kind == "word":
        return "A" * _LINE_LEN
    return (_B64_UNIT * (_LINE_LEN // len(_B64_UNIT) + 1))[:_LINE_LEN]


def _minified_line(blob: str) -> str:
    """
    Description: Embed the canary token near the end of a minified-JS-like line after a long blob.
    Parameters:
        blob: the long run placed before the assignment.
    Output: the line, with no assignment before the blob so every start position must be tried.
    """
    return f'{blob};var k="{_TOKEN}";'


def test_fr_core_07_feature_var_name_long_line():
    line = _minified_line(_long_runs("word"))
    start = time.perf_counter()
    assert feature_var_name(line, _TOKEN) == "k"
    assert time.perf_counter() - start < _TIME_BOUND_SECONDS


def test_fr_core_07_feature_var_name_normal_lines_unchanged():
    # Pinned to the behavior before the window was added; the model features depend on it.
    assert feature_var_name('api_key = "abc123secretvalue"', "abc123secretvalue") == "api_key"
    assert feature_var_name('DB_PASSWORD: "Wint3r#9"', "Wint3r#9") == "DB_PASSWORD"
    assert (
        feature_var_name('  DB_PASSWORD: "Wint3r#9"', "Wint3r#9") is None
    )  # `^` anchors at line start
    assert feature_var_name('connect("abc123secretvalue")', "abc123secretvalue") is None
    assert feature_var_name("TOKEN=abc123secretvalue", "abc123secretvalue") == "TOKEN"


@pytest.mark.parametrize("kind", ["word", "base64"])
def test_fr_core_07_extract_features_long_line(kind):
    line = _minified_line(_long_runs(kind))
    finding = Finding(
        type="ENTROPY_CANDIDATE",
        snippet="minified",
        evidence=EvidenceType.ENTROPY,
        severity=Severity.HIGH,
        token=_TOKEN,
        line=1,
    )
    start = time.perf_counter()
    context = extract_context(line, 1)
    extract_features(finding, context)
    assert time.perf_counter() - start < _TIME_BOUND_SECONDS


def _spaces_then_assignment(count: int) -> str:
    """
    Description: Build a whitespace run followed by an assignment, to stress whitespace-adjacent regexes.
    Parameters:
        count: number of spaces before the assignment.
    Output: the line.
    """
    return " " * count + f'var k="{_TOKEN}";'


@pytest.mark.parametrize("count", [5_000, 50_000])
def test_fr_core_07_java_func_pattern_whitespace_run(count):
    line = _spaces_then_assignment(count)
    finding = Finding(
        type="ENTROPY_CANDIDATE",
        snippet="whitespace",
        evidence=EvidenceType.ENTROPY,
        severity=Severity.HIGH,
        token=_TOKEN,
        line=1,
    )
    start = time.perf_counter()
    context = extract_context(line, 1)
    extract_features(finding, context)
    assert time.perf_counter() - start < _TIME_BOUND_SECONDS


_OLD_JAVA_FUNC_PATTERN = re.compile(
    r"^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(", re.M
)
_NEW_JAVA_FUNC_PATTERN = re.compile(
    r"^[ \t]*(?:(?:public|private|protected)[ \t]+)?(?:static[ \t]+)?\w+[ \t]+\w+[ \t]*\(", re.M
)

_JAVA_FUNC_SNIPPETS = [
    "public static void main(String[] args) {",
    "private int foo (int a) {",
    "protected final void run() {",
    "String bar(int x) {",
    "static int x() {",
    "int total(",
    "    public void indented() {",
    "\tprivate String tabbed(String s) {",
    "\t\tstatic int deep(",
    "\n\n  public int leadingBlank() {",
    "x = 1\npublic void second() {\n}",
    "return a + b(",
    "x = foo(1)",
    "foo(1, 2)",
    "def helper(self, x):",
    "    def method(self):",
    "function render(props) {",
    "func Handle(w http.ResponseWriter) {",
    "const f = (a) => a",
    "api_key = 'abc'\nconnect(api_key)",
    "if (x > 1) {",
    "new Thing(a)",
    "public\nvoid run(",
    "public static\nvoid run(",
    "\n\n\n  \n    \nint x(",
]


@pytest.mark.parametrize("text", _JAVA_FUNC_SNIPPETS)
def test_fr_core_07_java_func_pattern_same_results(text):
    assert bool(_NEW_JAVA_FUNC_PATTERN.search(text)) == bool(_OLD_JAVA_FUNC_PATTERN.search(text))


# Known, accepted semantic difference: the old pattern let `\s` span a newline, so a type and name
# (or name and `(`) split across lines counted as a function definition. The linear pattern is
# line-bounded; `\s` across newlines is what made it cubic. Real Java signatures with the whole
# `type name(` on one line, and any such line inside a multi-line context, match identically.
@pytest.mark.parametrize("text", ["int\nfoo(", "void run\n("])
def test_fr_core_07_java_func_pattern_newline_split_differs(text):
    assert _OLD_JAVA_FUNC_PATTERN.search(text)
    assert not _NEW_JAVA_FUNC_PATTERN.search(text)
