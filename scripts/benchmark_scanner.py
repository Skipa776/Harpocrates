#!/usr/bin/env python3
"""Reproducible scanner benchmarks with explicit ONNX execution evidence.

The harness reports measurements as JSON so performance claims can be compared
across commits without parsing human-formatted CLI output. It never serializes
finding tokens or snippets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.core.scanner import (
    DEFAULT_IGNORE_PATTERNS,
    ScanEngine,
    scan_directory,
)
from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.onnx_verifier import OnnxVerifier

SOURCE_VIEW_IGNORES = {
    ".coverage",
    ".gstack",
    "artifacts",
    "comparison",
    "data",
    "harpocrates.egg-info",
    "images",
    "models",
}


def summarize(samples_ms: list[float]) -> dict[str, float | int]:
    """Return deterministic latency summary statistics for millisecond samples."""
    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(samples_ms)

    def percentile(value: float) -> float:
        index = max(0, math.ceil(value * len(ordered)) - 1)
        return ordered[index]

    return {
        "iterations": len(ordered),
        "min_ms": min(ordered),
        "mean_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": percentile(0.95),
        "max_ms": max(ordered),
    }


def _time_calls(
    operation: Callable[[], Any],
    *,
    iterations: int,
    warmups: int,
    validate_result: Callable[[Any], None] | None = None,
) -> tuple[list[float], Any]:
    for _ in range(warmups):
        operation()
    samples = []
    last_result = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        last_result = operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        if validate_result is not None:
            validate_result(last_result)
    return samples, last_result


class _CountingSession:
    """Transparent ONNX Runtime session proxy that proves inference occurred."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.run_count = 0

    def get_inputs(self):
        return self._session.get_inputs()

    def run(self, *args, **kwargs):
        self.run_count += 1
        return self._session.run(*args, **kwargs)


def benchmark_onnx(*, iterations: int, warmups: int, candidate_count: int = 64) -> dict[str, Any]:
    """Measure feature extraction plus a real batched ONNX inference call."""
    verifier = OnnxVerifier(lazy_load=False)
    counting_session = _CountingSession(verifier._session)
    verifier._session = counting_session

    candidates = []
    for index in range(candidate_count):
        token = f"aB3dEfGhIjKlMnOpQrStUvWxYz{index:06d}"
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet=f'api_secret = "{token}"',
            evidence=EvidenceType.ENTROPY,
            token=token,
            confidence=0.7,
        )
        context = CodeContext(
            line_content=f'api_secret = "{token}"',
            file_path="benchmark_config.py",
        )
        candidates.append((finding, context))

    samples, results = _time_calls(
        lambda: verifier.verify_batch(candidates),
        iterations=iterations,
        warmups=warmups,
    )
    expected_runs = iterations + warmups
    if counting_session.run_count != expected_runs:
        raise RuntimeError(
            "ONNX benchmark did not execute the expected number of sessions: "
            f"expected {expected_runs}, got {counting_session.run_count}"
        )
    if len(results) != candidate_count:
        raise RuntimeError("ONNX benchmark returned an incomplete result batch")

    return {
        "candidate_count": candidate_count,
        "onnx_session_runs": counting_session.run_count,
        "feature_count": verifier.input_feature_count,
        "timing": summarize(samples),
    }


def benchmark_directory(
    root: Path,
    *,
    engine: ScanEngine,
    iterations: int,
    warmups: int,
    max_file_size: int,
    ignore_patterns: set[str],
) -> dict[str, Any]:
    """Measure full traversal, scan, transport, and Python mapping."""
    root_resolved = root.resolve()
    expected_signature: tuple[int, int, int, str] | None = None

    def result_signature(sample: Any) -> tuple[int, int, int, str]:
        fingerprints = []
        for finding in sample.findings:
            finding_path = Path(finding.file).resolve() if finding.file else root_resolved
            try:
                normalized_path = str(finding_path.relative_to(root_resolved))
            except ValueError:
                normalized_path = str(finding_path)
            fingerprints.append(
                (
                    normalized_path,
                    finding.line,
                    finding.type,
                    finding.evidence.value,
                    finding.token,
                )
            )
        fingerprint_payload = json.dumps(
            sorted(fingerprints), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return (
            sample.scanned_files,
            sample.total_lines,
            len(sample.findings),
            hashlib.sha256(fingerprint_payload).hexdigest(),
        )

    def validate_sample(sample: Any) -> None:
        nonlocal expected_signature
        if sample.errors:
            raise RuntimeError(
                f"{engine} directory benchmark failed: " + "; ".join(sample.errors)
            )
        signature = result_signature(sample)
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise RuntimeError(
                f"{engine} directory benchmark produced inconsistent measured results"
            )

    samples, result = _time_calls(
        lambda: scan_directory(
            root,
            engine=engine,
            max_file_size=max_file_size,
            ignore_patterns=ignore_patterns,
        ),
        iterations=iterations,
        warmups=warmups,
        validate_result=validate_sample,
    )
    scanned_files, total_lines, finding_count, fingerprint = result_signature(result)
    effective_ignores = DEFAULT_IGNORE_PATTERNS | ignore_patterns
    return {
        "engine": engine,
        "root": str(root.resolve()),
        "scanned_files": scanned_files,
        "total_lines": total_lines,
        "finding_count": finding_count,
        "finding_fingerprint_sha256": fingerprint,
        "error_count": len(result.errors),
        "ignore_patterns": sorted(effective_ignores),
        "timing": summarize(samples),
    }


def validate_directory_parity(python_result: dict[str, Any], rust_result: dict[str, Any]) -> None:
    """Reject performance comparisons that did not scan equivalent work."""
    for field in (
        "scanned_files",
        "total_lines",
        "finding_count",
        "finding_fingerprint_sha256",
    ):
        if python_result[field] != rust_result[field]:
            raise RuntimeError(
                "directory benchmark engine mismatch for "
                f"{field}: python={python_result[field]}, rust={rust_result[field]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--onnx-candidates", type=int, default=64)
    parser.add_argument("--max-size-mb", type=int, default=10)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument(
        "--full-tree",
        action="store_true",
        help=("Disable benchmark-only source-view excludes; scanner defaults still apply"),
    )
    args = parser.parse_args()

    if args.iterations < 1 or args.warmups < 0 or args.onnx_candidates < 1:
        parser.error("iterations/candidates must be positive and warmups non-negative")

    output: dict[str, Any] = {
        "schema_version": 1,
        "onnx": benchmark_onnx(
            iterations=args.iterations,
            warmups=args.warmups,
            candidate_count=args.onnx_candidates,
        ),
    }
    max_file_size = args.max_size_mb * 1024 * 1024
    ignore_patterns = set() if args.full_tree else SOURCE_VIEW_IGNORES
    python_directory = None
    if not args.skip_python:
        python_directory = benchmark_directory(
            args.root,
            engine="python",
            iterations=args.iterations,
            warmups=args.warmups,
            max_file_size=max_file_size,
            ignore_patterns=ignore_patterns,
        )
        output["python_directory"] = python_directory
    rust_directory = benchmark_directory(
        args.root,
        engine="rust",
        iterations=args.iterations,
        warmups=args.warmups,
        max_file_size=max_file_size,
        ignore_patterns=ignore_patterns,
    )
    output["rust_directory"] = rust_directory
    if python_directory is not None:
        validate_directory_parity(python_directory, rust_directory)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
