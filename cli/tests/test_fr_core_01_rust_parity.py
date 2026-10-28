"""FR-CORE-01: the Rust core gives the same regex + entropy findings as the Python engine.

Skips when the binary isn't built. CI sets HARPOCRATES_REQUIRE_RUST=1 (HARPO_REQUIRE_RUST works too) so a
missing binary fails instead of silently testing Python twice.
"""

import dataclasses
import importlib
import json
import math
import random
import string
import subprocess
import sys
import types
from pathlib import Path

import pytest

from Harpocrates import envvars
from Harpocrates.core import detector
from Harpocrates.core.result import Finding
from Harpocrates.core.rust_backend import RustScannerBackend
from Harpocrates.core.scanner import scan_directory

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "samples" / "samples"
READMES = ROOT / "docs" / "samples" / "readme_examples.jsonl"

_ALNUM = string.ascii_letters + string.digits
_TOKEN_CHARS = _ALNUM + "_-"
_UPPER_ALNUM = string.ascii_uppercase + string.digits
_HEX = "0123456789abcdef"
_SYMBOL_SETS = ("_PASSWORD_SYMBOLS", "_SOUP_SYMBOLS")
_FIELDS = [f.name for f in dataclasses.fields(Finding) if f.name not in ("entropy", "confidence")]


@pytest.fixture(scope="module")
def rust_available():
    if RustScannerBackend.discover() is None:
        if envvars.getenv("REQUIRE_RUST") == "1":
            pytest.fail("HARPO_REQUIRE_RUST=1 (or HARPOCRATES_REQUIRE_RUST=1) but the Rust scanner binary was not found")
        pytest.skip("Rust scanner binary not built")


@pytest.fixture(scope="module")
def native():
    try:
        return importlib.import_module("harpo_native")
    except ImportError:
        if envvars.getenv("REQUIRE_NATIVE") == "1":
            pytest.fail("HARPO_REQUIRE_NATIVE=1 (or HARPOCRATES_REQUIRE_NATIVE=1) but harpo_native is not installed")
        pytest.skip("harpo_native is not installed (pip install ./core)")


def _keys(result):
    return sorted((Path(f.file).name, f.line, f.type, f.token) for f in result.findings)


def _assert_same(python, other, fields):
    """Assert two engines' candidates agree on these fields, with entropy and confidence within 1e-9."""

    def row(f):
        return tuple(getattr(f, n) for n in fields)

    assert [row(f) for f in python] == [row(f) for f in other]
    for a, b in zip(python, other):
        assert math.isclose(a.entropy, b.entropy, abs_tol=1e-9)
        assert math.isclose(a.confidence, b.confidence, abs_tol=1e-9)


def _same_candidates(path, monkeypatch):
    """Assert both engines return the same raw candidates for one file, and return the Python ones."""
    monkeypatch.setattr(detector, "_native", None)
    py = detector._collect_file_findings(path)
    rs = RustScannerBackend.discover(required=True)._collect_file(path, memory_cap=None)
    _assert_same(py, rs, [n for n in _FIELDS if n != "file"])
    return py


def _rule_files(tmp_path):
    """Write the fixture files that cover every regex and entropy rule with its edge cases; return their paths."""
    rng = random.Random(1)

    def pick(alphabet, n):
        return "".join(rng.choice(alphabet) for _ in range(n))

    def digits(n):
        return pick(string.digits, n)

    def alnum(n):
        return pick(_ALNUM, n)

    def token(n):
        return pick(_TOKEN_CHARS, n)

    def lower_hex(n):
        return pick(_HEX, n)

    def upper(n):
        return pick(_UPPER_ALNUM, n)

    def pw():
        return pick(string.ascii_letters, 6) + digits(3) + "!"

    soup = list(pick(string.ascii_letters, 4) + digits(2) + "!@#" + pick(string.ascii_letters, 3) + digits(2) + "(;)")
    rng.shuffle(soup)
    soup_text = "".join(soup)

    rules = [
        f'url = "https://api.telegram.org/bot{digits(10)}:{token(35)}/getMe"',
        f'tg = "{digits(9)}:{token(36)}"',
        f'tg = "{digits(11)}:{token(35)}"',
        f"x {digits(8)}:{pick(string.ascii_letters, 27)}{digits(8)}:{token(35)} y",
        # The first token's tail runs past 35 characters, so its lookahead fails; the later one inside it matches.
        f"y {digits(8)}:{pick(string.ascii_letters, 30)}A{digits(9)}:{token(35)} z",
        f'k = "sk-{alnum(48)}-{alnum(5)}"',
        f'k = "sk-{alnum(10)}_{alnum(10)}"',
        f'k = "sk-proj-{alnum(30)}"',
        f'mypassword = "{pw()}"',
        f'db_password = "{pw()}"',
        f'password_{"x" * 45} = "{pw()}"',
        f'dsn = "postgresql://app:{pw()}@db.internal:5432/orders"',
        f"see xpostgres://app:{pw()}@db/x",
        f'conn = "Server=db;Database=app;Password={pw()};"',
        f'fetch("https://api.example.com/v1/items?token={alnum(24)}&page=2")',
        f"sig = {lower_hex(32)}",
        f"sig64 = {lower_hex(64)}",
        f"long_hex = {lower_hex(70)}",
        f'call("{pw()}")',
        f'call("{pw()}{pick(string.ascii_letters, 54)}")',  # a quoted literal of exactly 64 characters, the limit
        f'headers = {{"Authorization": "Bearer {alnum(30)}"}}',
        f"X-Deploy-Key: {alnum(20)}",
        f"value = {soup_text}",
        f'dsn = "https://{lower_hex(32)}@o123.ingest.sentry.io/456"',
        f"see https://example.com/files/{pw()}{pw()}/download).",
        f'cb = "https://example.com/cb?state={pw()}{pw()}&code=abc"',
        f'item = "https://example.com/items/{lower_hex(8)}-{lower_hex(4)}-{lower_hex(4)}-{lower_hex(4)}-{lower_hex(12)}"',
        f'# password = "{pw()}"',
        "/*",
        f'   token = "{pw()}"',
        "*/",
        f'mot_de_passe = "é{pw()}ü"',
        f'clé: "{pw()}"',
        f'{"a = 1; " * 80}password = "{pw()}"',
        f'API = "{alnum(40)}"',
        'aws = "' + "AK" + "IA" + upper(16) + '"',
        "",
        "   ",
    ]
    files = {
        "rules.py": rules,
        "app.env": [
            f"DATABASE_URL=postgres://app:{pw()}@db/orders",
            f"SECRET_TOKEN={alnum(24)}",
            "HOST=localhost",
            f'API_KEY="{pw()}"',
            f"# comment {alnum(30)}",
        ],
        "settings.yaml": [
            f"api_key: {pw()}{pw()}",
            f"password: {pw()}",
            f"auth_token: '{alnum(20)}'",
        ],
        "page.html": [
            f'<div data-x="{alnum(40)}">',
            f'password = "{pw()}"',
        ],
        "cert.txt": [pick(_ALNUM + "+/", 64)],
    }

    paths = []
    for name, lines in files.items():
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def test_fr_core_01_rust_matches_python(rust_available):
    rust = scan_directory(SAMPLES, engine="rust")
    python = scan_directory(SAMPLES, engine="python")
    assert rust.errors == []  # Rust scan failure or per-file read errors
    assert rust.findings, "samples should contain findings"
    assert _keys(rust) == _keys(python)


def test_fr_core_01_rust_patterns_are_the_python_patterns(rust_available):
    """The Rust core matches with the Python patterns, so editing one without the other fails here."""
    executable = str(RustScannerBackend.discover(required=True).executable)
    out = subprocess.run([executable, "patterns"], capture_output=True, text=True, check=True)
    rust = [
        (name, "".join(sorted(text)) if name in _SYMBOL_SETS else text)
        for name, text in json.loads(out.stdout)
    ]
    assert rust == detector.rule_patterns()


def test_fr_core_01_every_rule_gives_the_same_candidates(rust_available, tmp_path, monkeypatch):
    """Every regex and entropy rule, with its edge cases, gives the same raw candidates on both engines."""
    findings = []
    for path in _rule_files(tmp_path):
        findings.extend(_same_candidates(path, monkeypatch))

    types_found = {f.type for f in findings}
    assert {
        "TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY_LEGACY", "AWS_ACCESS_KEY_ID", "ENTROPY_CANDIDATE", "ML_CANDIDATE",
        "ENV_ASSIGNMENT",
    } <= types_found
    assert sum(f.type == "ML_CANDIDATE" for f in findings) >= 15


def test_fr_core_01_samples_give_the_same_candidates(rust_available, monkeypatch):
    """The README's samples give the same raw candidates on both engines."""
    paths = sorted(p for p in SAMPLES.rglob("*") if p.is_file())
    assert paths, "docs/samples/samples should contain files"
    for path in [*paths, READMES]:
        _same_candidates(path, monkeypatch)


def test_fr_core_01_native_rules_give_the_same_candidates(native, tmp_path, monkeypatch):
    """The harpo_native rules give the same candidates as the Python rules on every fixture and README sample."""
    paths = [*_rule_files(tmp_path), *sorted(p for p in SAMPLES.rglob("*") if p.is_file()), READMES]
    for path in paths:
        monkeypatch.setattr(detector, "_native", None)
        python = detector._collect_file_findings(path)
        monkeypatch.setattr(detector, "_native", native)
        native_found = detector._collect_file_findings(path)
        _assert_same(python, native_found, _FIELDS)


def test_fr_core_01_native_is_used_only_with_the_same_rules(native, monkeypatch):
    """harpo_native is used only with exactly the Python rules; otherwise, or with HARPO_NATIVE=0, Python runs."""
    monkeypatch.delenv("HARPO_NATIVE", raising=False)
    monkeypatch.delenv("HARPOCRATES_NATIVE", raising=False)
    monkeypatch.setattr(detector, "_native", detector._UNSET)
    assert detector._native_module() is native

    monkeypatch.setattr(detector, "_native", detector._UNSET)
    monkeypatch.setenv("HARPO_NATIVE", "0")
    assert detector._native_module() is None

    monkeypatch.delenv("HARPO_NATIVE", raising=False)
    monkeypatch.setattr(detector, "_native", detector._UNSET)
    other_rules = types.SimpleNamespace(
        PROTOCOL_VERSION=native.PROTOCOL_VERSION, patterns=lambda: [("X", "y")], scan_line=native.scan_line,
    )
    monkeypatch.setitem(sys.modules, "harpo_native", other_rules)
    assert detector._native_module() is None

    monkeypatch.setattr(detector, "_native", detector._UNSET)
    other_protocol = types.SimpleNamespace(
        PROTOCOL_VERSION=native.PROTOCOL_VERSION + 1, patterns=native.patterns, scan_line=native.scan_line,
    )
    monkeypatch.setitem(sys.modules, "harpo_native", other_protocol)
    assert detector._native_module() is None


def test_fr_core_01_native_is_in_use_when_required():
    """With REQUIRE_NATIVE=1 (CI's Rust job), the harpo_native rules are the ones in use."""
    if envvars.getenv("REQUIRE_NATIVE") == "1":
        assert detector._native_module() is not None
    else:
        pytest.skip("HARPO_REQUIRE_NATIVE is not set")
