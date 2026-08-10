"""Behavioral checks for the scanner benchmark harness."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.benchmark_scanner as benchmark
from scripts.benchmark_scanner import (
    benchmark_directory,
    benchmark_onnx,
    summarize,
    validate_directory_parity,
)


def test_summarize_reports_stable_percentiles() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["iterations"] == 5
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == 5.0


def test_benchmark_onnx_proves_session_execution() -> None:
    result = benchmark_onnx(iterations=1, warmups=0, candidate_count=4)

    assert result["candidate_count"] == 4
    assert result["onnx_session_runs"] == 1
    assert result["timing"]["iterations"] == 1


def test_directory_benchmark_rejects_scan_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        benchmark,
        "scan_directory",
        lambda *args, **kwargs: SimpleNamespace(
            scanned_files=0,
            total_lines=0,
            findings=[],
            errors=["native process failed"],
        ),
    )

    with pytest.raises(RuntimeError, match="native process failed"):
        benchmark_directory(
            tmp_path,
            engine="rust",
            iterations=1,
            warmups=0,
            max_file_size=100,
            ignore_patterns=set(),
        )


def test_directory_benchmark_rejects_an_intermediate_scan_error(
    monkeypatch, tmp_path
) -> None:
    calls = 0

    def scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            scanned_files=1,
            total_lines=1,
            findings=[],
            errors=["intermittent native failure"] if calls == 1 else [],
        )

    monkeypatch.setattr(benchmark, "scan_directory", scan)

    with pytest.raises(RuntimeError, match="intermittent native failure"):
        benchmark_directory(
            tmp_path,
            engine="rust",
            iterations=2,
            warmups=0,
            max_file_size=100,
            ignore_patterns=set(),
        )


def test_directory_benchmark_rejects_engine_parity_mismatch() -> None:
    python_result = {
        "scanned_files": 10,
        "total_lines": 20,
        "finding_count": 3,
        "finding_fingerprint_sha256": "python",
    }
    rust_result = {
        "scanned_files": 10,
        "total_lines": 20,
        "finding_count": 2,
        "finding_fingerprint_sha256": "rust",
    }

    with pytest.raises(RuntimeError, match="finding_count"):
        validate_directory_parity(python_result, rust_result)


def test_directory_benchmark_rejects_same_count_different_findings() -> None:
    python_result = {
        "scanned_files": 10,
        "total_lines": 20,
        "finding_count": 3,
        "finding_fingerprint_sha256": "python",
    }
    rust_result = {
        "scanned_files": 10,
        "total_lines": 20,
        "finding_count": 3,
        "finding_fingerprint_sha256": "rust",
    }

    with pytest.raises(RuntimeError, match="finding_fingerprint_sha256"):
        validate_directory_parity(python_result, rust_result)


def test_directory_benchmark_rejects_line_accounting_mismatch() -> None:
    python_result = {
        "scanned_files": 10,
        "total_lines": 19,
        "finding_count": 3,
        "finding_fingerprint_sha256": "same",
    }
    rust_result = {
        "scanned_files": 10,
        "total_lines": 20,
        "finding_count": 3,
        "finding_fingerprint_sha256": "same",
    }

    with pytest.raises(RuntimeError, match="total_lines"):
        validate_directory_parity(python_result, rust_result)


def test_directory_benchmark_reports_effective_ignore_policy(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        benchmark,
        "scan_directory",
        lambda *args, **kwargs: SimpleNamespace(
            scanned_files=0,
            total_lines=0,
            findings=[],
            errors=[],
        ),
    )

    result = benchmark_directory(
        tmp_path,
        engine="python",
        iterations=1,
        warmups=0,
        max_file_size=100,
        ignore_patterns={"custom"},
    )

    assert ".git" in result["ignore_patterns"]
    assert "custom" in result["ignore_patterns"]
