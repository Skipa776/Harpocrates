"""FR-CORE-07: feature extraction runs in time linear in line length (minified JS, inline base64)."""

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
    assert feature_var_name('  DB_PASSWORD: "Wint3r#9"', "Wint3r#9") is None  # `^` anchors at line start
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
