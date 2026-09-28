"""Contract tests for the native scanner's Python boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.core.rust_backend import (
    PROTOCOL_VERSION,
    NativeBatchResult,
    NativeFileResult,
    RustScannerBackend,
    RustScannerError,
)


def _completed(payload: object) -> SimpleNamespace:
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
    verifier.verify_batch.return_value = [
        SimpleNamespace(
            is_secret=True,
            combined_confidence=0.91,
        )
    ]

    backend = RustScannerBackend(Path("/tmp/harpocrates-rust-scanner"))
    findings = backend.scan_file_with_ml(path, verifier=verifier, ml_threshold=0.5)

    verifier.verify_batch.assert_called_once()
    assert findings[0].evidence == EvidenceType.HYBRID
    assert findings[0].confidence == 0.91
    assert findings[0].category == "password"


def test_rust_backend_propagates_python_ml_failure(monkeypatch) -> None:
    """Native candidate generation must not disguise a broken ML verifier."""
    candidate = {
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
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed([candidate])),
    )
    verifier = Mock()
    verifier.verify_batch.side_effect = RuntimeError("broken model")

    backend = RustScannerBackend(Path("/tmp/harpocrates-rust-scanner"))

    with pytest.raises(RuntimeError, match="broken model"):
        backend.scan_text_with_ml('password="hunter2"', verifier=verifier)


def test_scan_file_with_ml_propagates_context_read_failure(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    candidate = Finding(
        type="ML_CANDIDATE",
        file=str(missing),
        line=1,
        snippet='password="hunter2"',
        token="hunter2",
        evidence=EvidenceType.ML,
    )
    backend = RustScannerBackend(Path("/tmp/native-scanner"))
    monkeypatch.setattr(backend, "_collect_file", Mock(return_value=[candidate]))

    with pytest.raises(RustScannerError, match="ML verification"):
        backend.scan_file_with_ml(missing, verifier=Mock())


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
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {
                "path": str(path),
                "findings": [
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
                ],
                "scanned": True,
                "line_count": 1,
                "bytes_scanned": 19,
                "error": None,
            }
        ],
    }
    run = Mock(return_value=_completed(payload))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)

    backend = RustScannerBackend(Path("/tmp/native-scanner"))
    findings = backend.scan_file(path, max_bytes=100)

    assert findings == []
    request = json.loads(run.call_args.kwargs["input"])
    assert request["files"][0]["max_bytes"] == 100


def test_rust_backend_scan_file_raises_on_native_file_error(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "missing.env"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {
                "path": str(path),
                "findings": [],
                "scanned": False,
                "line_count": 0,
                "bytes_scanned": 0,
                "error": "failed to read file",
            }
        ],
    }
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed(payload)),
    )

    with pytest.raises(RustScannerError, match="failed to read file"):
        RustScannerBackend(Path("/tmp/native-scanner")).scan_file(path)


def test_scan_files_starts_one_versioned_native_process(monkeypatch, tmp_path: Path) -> None:
    """A multi-file request pays native process startup only once."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("APP_NAME=A\n", encoding="utf-8")
    second.write_text("APP_NAME=B\n", encoding="utf-8")
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {
                "path": str(first),
                "findings": [],
                "scanned": True,
                "line_count": 1,
                "bytes_scanned": 11,
                "error": None,
            },
            {
                "path": str(second),
                "findings": [],
                "scanned": True,
                "line_count": 1,
                "bytes_scanned": 11,
                "error": None,
            },
        ],
    }
    run = Mock(return_value=_completed(response))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)

    batch = RustScannerBackend(Path("/tmp/native-scanner")).scan_files(
        [second, first], max_bytes=123
    )

    assert run.call_count == 1
    assert run.call_args.args[0][-1] == "scan-batch"
    request = json.loads(run.call_args.kwargs["input"])
    assert request == {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {"path": str(first), "max_bytes": 123},
            {"path": str(second), "max_bytes": 123},
        ],
    }
    assert [result.path for result in batch.files] == [first, second]


def test_scan_files_preserves_partial_errors(monkeypatch, tmp_path: Path) -> None:
    """A failed file must not discard successful files from the same batch."""
    good = tmp_path / "good.env"
    bad = tmp_path / "bad.env"
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {
                "path": str(bad),
                "findings": [],
                "scanned": False,
                "line_count": 0,
                "bytes_scanned": 0,
                "error": "permission denied",
            },
            {
                "path": str(good),
                "findings": [],
                "scanned": True,
                "line_count": 2,
                "bytes_scanned": 20,
                "error": None,
            },
        ],
    }
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed(response)),
    )

    batch = RustScannerBackend(Path("/tmp/native-scanner")).scan_files([bad, good])

    assert batch.files[0].error == "permission denied"
    assert not batch.files[0].scanned
    assert batch.files[1].scanned
    assert batch.scanned_files == 1
    assert batch.total_lines == 2


def test_scan_files_rejects_false_clean_metadata(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.env"
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {
                "path": str(path),
                "findings": [],
                "scanned": False,
                "line_count": 0,
                "bytes_scanned": 0,
                "error": None,
            }
        ],
    }
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed(response)),
    )

    with pytest.raises(RustScannerError, match="invalid batch metadata"):
        RustScannerBackend(Path("/tmp/native-scanner")).scan_files([path])


def test_discover_resolves_configured_relative_executable(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "scanner"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARPOCRATES_RUST_SCANNER", "scanner")

    backend = RustScannerBackend.discover(required=True)

    assert backend is not None
    assert backend.executable == executable.resolve()


def test_scan_files_rejects_wrong_protocol_version(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "config.txt"
    response = {"protocol_version": PROTOCOL_VERSION + 1, "files": []}
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed(response)),
    )

    backend = RustScannerBackend(Path("/tmp/native-scanner"))
    with pytest.raises(RustScannerError, match="protocol version"):
        backend.scan_files([path])


def test_scan_directory_sends_ignore_policy_to_rust(monkeypatch, tmp_path: Path) -> None:
    """Python owns policy while Rust owns traversal and pruning."""
    response = {"protocol_version": PROTOCOL_VERSION, "files": []}
    run = Mock(return_value=_completed(response))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)

    batch = RustScannerBackend(Path("/tmp/native-scanner")).scan_directory(
        tmp_path,
        recursive=False,
        max_file_size=456,
        ignore_patterns={"node_modules", "*.lock"},
    )

    assert batch.files == []
    assert run.call_args.args[0][-1] == "scan-directory"
    assert run.call_args.kwargs["timeout"] == 300.0
    request = json.loads(run.call_args.kwargs["input"])
    assert request == {
        "protocol_version": PROTOCOL_VERSION,
        "root": str(tmp_path.resolve()),
        "recursive": False,
        "max_file_size": 456,
        "ignore_patterns": ["*.lock", "node_modules"],
    }


def test_scan_directory_timeout_is_configurable(monkeypatch, tmp_path: Path) -> None:
    response = {"protocol_version": PROTOCOL_VERSION, "files": []}
    run = Mock(return_value=_completed(response))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)
    monkeypatch.setenv("HARPOCRATES_RUST_TIMEOUT_SECONDS", "42.5")

    RustScannerBackend(Path("/tmp/native-scanner")).scan_directory(
        tmp_path,
        recursive=True,
        max_file_size=1024,
        ignore_patterns=set(),
    )

    assert run.call_args.kwargs["timeout"] == 42.5


def test_scan_directory_rejects_path_outside_root(monkeypatch, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.env"
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            {
                "path": str(outside),
                "findings": [],
                "scanned": True,
                "line_count": 1,
                "bytes_scanned": 1,
                "error": None,
            }
        ],
    }
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed(response)),
    )
    verifier = Mock()

    with pytest.raises(RustScannerError, match="outside the scan root"):
        RustScannerBackend(Path("/tmp/native-scanner")).scan_directory_with_ml(
            tmp_path,
            verifier=verifier,
            recursive=True,
            max_file_size=100,
            ignore_patterns=set(),
        )

    verifier.verify_batch.assert_not_called()


def test_scan_directory_with_ml_bounds_verifier_batches(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "many.py"
    path.write_text('password="hunter2"\n', encoding="utf-8")
    candidates = [
        Finding(
            type="ML_CANDIDATE",
            file=str(path),
            line=1,
            snippet='password="hunter2"',
            token="hunter2",
            evidence=EvidenceType.ML,
        )
        for _ in range(1025)
    ]
    backend = RustScannerBackend(Path("/tmp/native-scanner"))
    monkeypatch.setattr(
        backend,
        "_collect_directory",
        Mock(
            return_value=NativeBatchResult(
                [
                    NativeFileResult(
                        path=path,
                        findings=candidates,
                        scanned=True,
                        line_count=1,
                        bytes_scanned=path.stat().st_size,
                    )
                ]
            )
        ),
    )
    verifier = Mock()
    batch_sizes = []

    def verify_batch(batch):
        batch_sizes.append(len(batch))
        return [SimpleNamespace(is_secret=True, combined_confidence=0.9) for _ in batch]

    verifier.verify_batch.side_effect = verify_batch

    result = backend.scan_directory_with_ml(
        tmp_path,
        verifier=verifier,
        recursive=True,
        max_file_size=100,
        ignore_patterns=set(),
    )

    assert batch_sizes == [1024, 1]
    assert len(result.findings) == 1025


def test_scan_directory_with_ml_batches_python_verification(monkeypatch, tmp_path: Path) -> None:
    """One Rust process and one Python ML batch handle the whole directory."""
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text('password="hunter2"\n', encoding="utf-8")
    second.write_text('token="letmein9"\n', encoding="utf-8")

    def candidate(path: Path, token: str, var_name: str) -> dict[str, object]:
        return {
            "path": str(path),
            "findings": [
                {
                    "type": "ML_CANDIDATE",
                    "line": 1,
                    "snippet": f'{var_name}="{token}"',
                    "token": token,
                    "token_start": len(var_name) + 2,
                    "token_end": len(var_name) + 2 + len(token),
                    "evidence": "ml",
                    "severity": "info",
                    "confidence": 0.5,
                    "entropy": 2.8,
                    "in_comment": None,
                    "var_name": var_name,
                }
            ],
            "scanned": True,
            "line_count": 1,
            "bytes_scanned": path.stat().st_size,
            "error": None,
        }

    response = {
        "protocol_version": PROTOCOL_VERSION,
        "files": [
            candidate(first, "hunter2", "password"),
            candidate(second, "letmein9", "token"),
        ],
    }
    run = Mock(return_value=_completed(response))
    monkeypatch.setattr("Harpocrates.core.rust_backend.subprocess.run", run)
    verifier = Mock()
    verifier.verify_batch.return_value = [
        SimpleNamespace(is_secret=True, combined_confidence=0.91),
        SimpleNamespace(is_secret=True, combined_confidence=0.92),
    ]

    batch = RustScannerBackend(Path("/tmp/native-scanner")).scan_directory_with_ml(
        tmp_path,
        verifier=verifier,
        recursive=True,
        max_file_size=1000,
        ignore_patterns=set(),
        ml_threshold=0.5,
    )

    assert run.call_count == 1
    verifier.verify_batch.assert_called_once()
    assert [finding.evidence for finding in batch.findings] == [
        EvidenceType.HYBRID,
        EvidenceType.HYBRID,
    ]


def test_invalid_finding_payload_does_not_echo_secret(monkeypatch) -> None:
    secret = "top-secret-value-that-must-not-leak"
    monkeypatch.setattr(
        "Harpocrates.core.rust_backend.subprocess.run",
        Mock(return_value=_completed([{"token": secret}])),
    )
    backend = RustScannerBackend(Path("/tmp/native-scanner"))

    with pytest.raises(RustScannerError) as exc_info:
        backend.scan_text("ignored")

    assert secret not in str(exc_info.value)
