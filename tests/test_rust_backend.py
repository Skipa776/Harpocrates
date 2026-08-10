"""Contract tests for the native scanner's Python boundary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Harpocrates.core.result import EvidenceType
from Harpocrates.core.rust_backend import RustScannerBackend, RustScannerError


def _completed(payload: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")


def test_rust_backend_maps_native_findings(monkeypatch) -> None:
    payload = [
        {
            "type": "GITHUB_PAT",
            "line": 2,
            "snippet": "token=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "token": "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "token_start": 6,
            "token_end": 46,
            "evidence": "regex",
            "severity": "critical",
            "confidence": 0.99,
            "entropy": 1.2,
            "in_comment": None,
            "var_name": None,
        }
    ]
    run = Mock(return_value=_completed(payload))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)

    backend = RustScannerBackend(Path("/tmp/harpocrates-rust-scanner"))
    findings = backend.scan_text("ignored")

    assert findings[0].type == "GITHUB_PAT"
    assert findings[0].evidence == EvidenceType.REGEX
    assert findings[0].category == "github_token"
    assert findings[0].token_start == 6
    assert run.call_args.kwargs["input"] == "ignored"


def test_rust_backend_keeps_python_ml_verifier(monkeypatch, tmp_path: Path) -> None:
    candidate = {
        "type": "ML_CANDIDATE",
        "line": 1,
        "snippet": 'password="hunter2"',
        "token": "hunter2",
        "token_start": 10,
        "token_end": 17,
        "evidence": "ml",
        "severity": "high",
        "confidence": 0.5,
        "entropy": 2.8,
        "in_comment": None,
        "var_name": "password",
    }
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed([candidate])),
    )
    path = tmp_path / "config.py"
    path.write_text('password="hunter2"\n', encoding="utf-8")
    verifier = Mock()
    verifier.verify.return_value = SimpleNamespace(
        is_secret=True,
        combined_confidence=0.91,
    )

    backend = RustScannerBackend(Path("/tmp/harpocrates-rust-scanner"))
    findings = backend.scan_file_with_ml(path, verifier=verifier, ml_threshold=0.5)

    verifier.verify.assert_called_once()
    assert findings[0].evidence == EvidenceType.HYBRID
    assert findings[0].confidence == 0.91
    assert findings[0].category == "password"


def test_discover_uses_configured_executable(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "native-scanner"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("HARPOCRATES_RUST_SCANNER", str(executable))
    monkeypatch.setattr("Harpocrates.core.rust_backend.shutil.which", lambda name: None)

    backend = RustScannerBackend.discover(required=True)

    assert backend is not None
    assert backend.executable == executable


def test_discover_required_reports_build_instruction(monkeypatch, tmp_path: Path) -> None:
    import Harpocrates.core.rust_backend as rust_backend

    monkeypatch.delenv("HARPOCRATES_RUST_SCANNER", raising=False)
    monkeypatch.setattr(rust_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(rust_backend, "__file__", str(tmp_path / "package" / "rust_backend.py"))

    with pytest.raises(RustScannerError, match="cargo build --release"):
        RustScannerBackend.discover(required=True)


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (SimpleNamespace(returncode=2, stdout="", stderr="native error"), "native error"),
        (SimpleNamespace(returncode=0, stdout="not-json", stderr=""), "invalid JSON"),
        (SimpleNamespace(returncode=0, stdout="{}", stderr=""), "result shape"),
    ],
)
def test_rust_backend_rejects_process_and_protocol_errors(
    monkeypatch, completed: SimpleNamespace, message: str
) -> None:
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=completed),
    )
    backend = RustScannerBackend(Path("/tmp/native-scanner"))

    with pytest.raises(RustScannerError, match=message):
        backend.scan_text("ignored")


def test_rust_backend_scan_file_filters_unverified_ml(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "config.py"
    path.write_text('password="hunter2"\n', encoding="utf-8")
    payload = [
        {
            "type": "ML_CANDIDATE",
            "line": 1,
            "snippet": 'password="hunter2"',
            "token": "hunter2",
            "token_start": 10,
            "token_end": 17,
            "evidence": "ml",
            "severity": "info",
            "confidence": 0.5,
            "entropy": 2.8,
            "in_comment": None,
            "var_name": "password",
        }
    ]
    run = Mock(return_value=_completed(payload))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)

    backend = RustScannerBackend(Path("/tmp/native-scanner"))
    findings = backend.scan_file(path, max_bytes=100)

    assert findings == []
    assert run.call_args.args[0][-2:] == ["--max-bytes", "100"]
