"""Acceptance tests for the M1 read tool: FR-READ-01, FR-READ-02, SEC-02, FR-CORE-02.

Every secret here is a generated canary from bench/canary_repo.py; none is a real credential.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from bench.canary_repo import build
from Harpocrates.core.classification import SECRET_TYPES, ViolationCategory, secret_type
from Harpocrates.read import Redactor

PLACEHOLDER = re.compile(r"<<HARPO:(" + "|".join(SECRET_TYPES) + r"):[0-9a-f]{4}>>")


def _leaks(output: str, canaries: list[dict]) -> list[str]:
    """Canary values (or any line of a multi-line canary) that appear in output."""
    leaked = []
    for c in canaries:
        parts = [p for p in c["value"].splitlines() if len(p) >= 8 and "-----" not in p]
        if any(p in output for p in parts or [c["value"]]):
            leaked.append(f'{c["file"]}:{c["kind"]}')
    return leaked


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    root = tmp_path_factory.mktemp("canary_repo")
    return root, build(root, seed=7)


def test_fr_read_01_safe_read_canary_repo_zero_leaks(repo) -> None:
    root, canaries = repo
    redactor = Redactor()
    files = sorted({c["file"] for c in canaries})
    for rel in files:
        out = redactor.safe_read(str(root / rel))
        assert not _leaks(out, canaries), f"{rel} leaked through safe_read"
        assert PLACEHOLDER.search(out), f"{rel} has canaries but no placeholder"


def test_fr_read_01_safe_grep_canary_repo_zero_leaks(repo) -> None:
    root, canaries = repo
    redactor = Redactor()
    for pattern in [".", "=", r"(?i)key|token|secret|password|://", "environment"]:
        out = redactor.safe_grep(pattern, str(root))
        assert out, f"{pattern!r} matched nothing"
        assert not _leaks(out, canaries), f"safe_grep {pattern!r} leaked"


def test_fr_read_01_grep_is_not_an_oracle(repo) -> None:
    """Searching for a secret's own value finds nothing: grep runs over redacted content."""
    root, canaries = repo
    redactor = Redactor()
    value = next(c["value"] for c in canaries if c["kind"] == "github_token")
    assert redactor.safe_grep(re.escape(value[:12]), str(root)) == ""


def test_fr_read_01_line_ranges_keep_numbering(tmp_path: Path) -> None:
    pem = "\n".join(["-----BEGIN RSA PRIVATE KEY-----", *["QUJD" * 16] * 3, "-----END RSA PRIVATE KEY-----"])
    path = tmp_path / "key_then_code.py"
    path.write_text(f"KEY = '''\n{pem}\n'''\nprint('line 8')\nprint('line 9')\n")
    out = Redactor().safe_read(str(path), start_line=8, end_line=9)
    assert out.splitlines()[:2] == ["     8\tprint('line 8')", "     9\tprint('line 9')"]
    whole = Redactor().safe_read(str(path))
    assert "QUJD" not in whole
    assert whole.count("<<HARPO:private_key:") == 5  # one placeholder per key line, numbering unchanged


def test_fr_read_01_ml_failure_still_redacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the ML verifier fails, every candidate is redacted (fail closed), never passed through."""
    import Harpocrates.read as read

    def broken(*_a, **_k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(read, "_apply_ml_verification", broken)
    path = tmp_path / "settings.py"
    path.write_text('DB_PASSWORD = "Kq7#vR2pLm9xTz4w"\n')
    out = Redactor().safe_read(str(path))
    assert "Kq7#vR2pLm9xTz4w" not in out


def test_fr_read_01_refuses_binary(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00\x01\x02" * 100)
    with pytest.raises(ValueError, match="binary"):
        Redactor().safe_read(str(path))


def test_fr_read_02_same_secret_same_placeholder(tmp_path: Path) -> None:
    token = "ghp_" + "Zx81kQ" * 6
    (tmp_path / "a.py").write_text(f'TOKEN = "{token}"\n')
    (tmp_path / "b.env").write_text(f"GITHUB_TOKEN={token}\n")
    redactor = Redactor()
    first = PLACEHOLDER.search(redactor.safe_read(str(tmp_path / "a.py")))
    second = PLACEHOLDER.search(redactor.safe_read(str(tmp_path / "b.env")))
    assert first and second
    assert first.group(0) == second.group(0)
    assert first.group(1) == "api_token"


def test_sec_02_session_keys_differ() -> None:
    a, b = Redactor(), Redactor()
    assert a._key != b._key and len(a._key) == 32
    values = [f"value-{i}" for i in range(8)]
    assert [a.placeholder(v, "generic") for v in values] != [b.placeholder(v, "generic") for v in values]


def test_fr_core_02_every_category_has_a_secret_type() -> None:
    assert set(SECRET_TYPES) == {"cloud_key", "api_token", "db_credential", "private_key", "password", "generic"}
    for category in ViolationCategory:
        assert secret_type(category.value) in SECRET_TYPES
    assert secret_type(None) == "generic"
    assert secret_type("aws_key") == "cloud_key"
    assert secret_type("connection_string") == "db_credential"


def test_fr_core_06_gate_layer_threshold() -> None:
    import json

    from Harpocrates.ml.onnx_verifier import MODEL_CONFIG_PATH, OnnxModelSchemaError, OnnxVerifier

    thresholds = json.loads(MODEL_CONFIG_PATH.read_text())["thresholds"]
    gate, commit = OnnxVerifier(layer="gate"), OnnxVerifier()
    gate._ensure_loaded()
    commit._ensure_loaded()
    assert gate._threshold_low == thresholds["gate"]
    assert commit._threshold_low == thresholds["commit"]
    assert thresholds["gate"] < thresholds["commit"]  # gate is the recall-first layer
    with pytest.raises(OnnxModelSchemaError, match="thresholds.nope"):
        OnnxVerifier(layer="nope")._ensure_loaded()


def test_fr_read_01_mcp_server_exposes_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from Harpocrates.mcp import server

    (tmp_path / "a.env").write_text("NPM_TOKEN=npm_" + "Qw3rTy" * 6 + "\n")
    assert "<<HARPO:api_token:" in server.safe_read(str(tmp_path / "a.env"))
    assert "a.env:1:NPM_TOKEN=<<HARPO:" in server.safe_grep("NPM", str(tmp_path))


def test_fr_read_01_pathological_line_is_linear(tmp_path: Path) -> None:
    """A long `token_token_...` line once took quadratic time in the credential-name regexes."""
    import time

    path = tmp_path / "min.js"
    path.write_text("token_" * 20000 + "\n")
    start = time.perf_counter()
    Redactor().safe_read(str(path))
    assert time.perf_counter() - start < 5


def test_fr_read_01_short_prefixed_token_not_shrunk() -> None:
    import Harpocrates.read as read

    assert read._NAME_PREFIX.sub("", "abcd==", count=1) == "="  # why the length guard exists
    text = "a = 1\nb = 2\n"
    assert Redactor().redact(text) == text
