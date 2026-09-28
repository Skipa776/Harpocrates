"""Python boundary for the native Harpocrates scanning engine.

Rust owns deterministic regex matching, entropy calculation, and candidate
generation. Python remains authoritative for category inference and optional
ML verification so existing model artifacts and verifier APIs stay unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from Harpocrates.core.classification import infer_category
from Harpocrates.core.detector import (
    _apply_ml_verification,
    _apply_ml_verification_with_contexts,
    _prepare_ml_context_from_lines,
    _severity_from_classification,
    _severity_from_entropy,
)
from Harpocrates.core.result import EvidenceType, Finding, Severity

if TYPE_CHECKING:
    from Harpocrates.ml.context import CodeContext
    from Harpocrates.ml.verifier import Verifier

_BINARY_ENV = "HARPOCRATES_RUST_SCANNER"
_BINARY_NAME = "harpocrates-rust-scanner"
_DIRECTORY_TIMEOUT_ENV = "HARPOCRATES_RUST_TIMEOUT_SECONDS"
_DEFAULT_DIRECTORY_TIMEOUT_SECONDS = 300.0
PROTOCOL_VERSION = 1
_ML_BATCH_SIZE = 1024


class RustScannerError(RuntimeError):
    """Raised when the native scanner cannot be located or executed."""


@dataclass(frozen=True)
class NativeFileResult:
    """Result and accounting metadata for one native file scan."""

    path: Path
    findings: list[Finding]
    scanned: bool
    line_count: int
    bytes_scanned: int
    error: Optional[str] = None


@dataclass(frozen=True)
class NativeBatchResult:
    """Validated aggregate returned by the versioned native protocol."""

    files: list[NativeFileResult]

    @property
    def findings(self) -> list[Finding]:
        return [finding for result in self.files for finding in result.findings]

    @property
    def scanned_files(self) -> int:
        return sum(result.scanned for result in self.files)

    @property
    def total_lines(self) -> int:
        return sum(result.line_count for result in self.files if result.scanned)

    @property
    def errors(self) -> list[str]:
        return [
            f"{result.path}: {result.error}" for result in self.files if result.error is not None
        ]


class RustScannerBackend:
    """Invoke the native scanner and map its candidates to Python findings."""

    def __init__(self, executable: str | Path) -> None:
        self.executable = Path(executable)

    @classmethod
    def discover(cls, *, required: bool = False) -> Optional["RustScannerBackend"]:
        """Find an installed binary or a binary built in this source checkout."""
        candidates: list[Path] = []
        configured = os.environ.get(_BINARY_ENV)
        if configured:
            candidates.append(Path(configured).expanduser())

        installed = shutil.which(_BINARY_NAME)
        if installed:
            candidates.append(Path(installed))

        repository_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            repository_root / "rust_scanner" / "target" / profile / _BINARY_NAME
            for profile in ("release", "debug")
        )

        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return cls(candidate.resolve())

        if required:
            raise RustScannerError(
                "Rust scanner binary not found. Run "
                "'cargo build --release --manifest-path rust_scanner/Cargo.toml' "
                f"or set {_BINARY_ENV}."
            )
        return None

    def scan_text(self, text: str, *, file: Optional[str] = None) -> list[Finding]:
        """Scan text natively, excluding unverified ML-only candidates."""
        findings = self._collect_text(text, file=file)
        return [finding for finding in findings if finding.evidence != EvidenceType.ML]

    def scan_text_with_ml(
        self,
        text: str,
        verifier: "Verifier",
        *,
        file: Optional[str] = None,
        ml_threshold: float = 0.5,
    ) -> list[Finding]:
        """Generate candidates in Rust and verify non-regex candidates in Python."""
        findings = self._collect_text(text, file=file)
        return self._verify_candidates(findings, text, verifier, ml_threshold)

    def scan_file(
        self,
        path: str | Path,
        *,
        max_bytes: Optional[int] = None,
    ) -> list[Finding]:
        """Scan a file natively, excluding unverified ML-only candidates."""
        batch = self.scan_files([Path(path)], max_bytes=max_bytes)
        if len(batch.files) != 1:
            raise RustScannerError("Rust scanner returned an invalid single-file result")
        result = batch.files[0]
        if not result.scanned or result.error is not None:
            raise RustScannerError(result.error or f"Rust scanner did not scan {result.path}")
        return result.findings

    def scan_files(
        self,
        paths: list[str | Path],
        *,
        max_bytes: Optional[int] = None,
    ) -> NativeBatchResult:
        """Scan files in one versioned native process, sorted deterministically."""
        ordered_paths = sorted(Path(path) for path in paths)
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "files": [{"path": str(path), "max_bytes": max_bytes} for path in ordered_paths],
        }
        payload = self._run(
            [str(self.executable), "scan-batch"],
            input_text=json.dumps(request),
        )
        return self._map_batch(payload, ordered_paths)

    def scan_directory(
        self,
        root: str | Path,
        *,
        recursive: bool,
        max_file_size: int,
        ignore_patterns: set[str],
    ) -> NativeBatchResult:
        """Traverse and scan a directory natively with Python-owned policy."""
        return self._collect_directory(
            root,
            recursive=recursive,
            max_file_size=max_file_size,
            ignore_patterns=ignore_patterns,
            include_ml_candidates=False,
        )

    def scan_directory_with_ml(
        self,
        root: str | Path,
        verifier: "Verifier",
        *,
        recursive: bool,
        max_file_size: int,
        ignore_patterns: set[str],
        ml_threshold: float = 0.5,
    ) -> NativeBatchResult:
        """Traverse once in Rust and verify candidates in bounded Python batches."""
        batch = self._collect_directory(
            root,
            recursive=recursive,
            max_file_size=max_file_size,
            ignore_patterns=ignore_patterns,
            include_ml_candidates=True,
        )
        pending: list[tuple[Finding, "CodeContext"]] = []
        verified_by_file: dict[str, list[Finding]] = {}

        def flush_pending() -> None:
            if not pending:
                return
            verified = _apply_ml_verification_with_contexts(
                pending,
                verifier=verifier,
                ml_threshold=ml_threshold,
            )
            for finding in verified:
                verified_by_file.setdefault(finding.file or "", []).append(finding)
            pending.clear()

        for file_result in batch.files:
            candidates = [
                finding
                for finding in file_result.findings
                if finding.evidence != EvidenceType.REGEX
            ]
            if not candidates:
                continue
            try:
                with file_result.path.open("r", encoding="utf-8", errors="ignore") as stream:
                    content = stream.read(max_file_size)
            except OSError as exc:
                raise RustScannerError(
                    f"failed to read {file_result.path} for ML verification: {exc}"
                ) from exc
            lines = content.splitlines()
            for finding in candidates:
                pending.append((finding, _prepare_ml_context_from_lines(finding, lines)))
                if len(pending) >= _ML_BATCH_SIZE:
                    flush_pending()
        flush_pending()

        files = []
        for file_result in batch.files:
            regex_findings = [
                finding
                for finding in file_result.findings
                if finding.evidence == EvidenceType.REGEX
            ]
            files.append(
                replace(
                    file_result,
                    findings=regex_findings + verified_by_file.get(str(file_result.path), []),
                )
            )
        return NativeBatchResult(files)

    def _collect_directory(
        self,
        root: str | Path,
        *,
        recursive: bool,
        max_file_size: int,
        ignore_patterns: set[str],
        include_ml_candidates: bool,
    ) -> NativeBatchResult:
        root_path = Path(root).resolve()
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "root": str(root_path),
            "recursive": recursive,
            "max_file_size": max_file_size,
            "ignore_patterns": sorted(ignore_patterns),
        }
        payload = self._run(
            [str(self.executable), "scan-directory"],
            input_text=json.dumps(request),
            timeout=self._directory_timeout(),
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            raise RustScannerError("Rust scanner returned an invalid directory result")
        paths = []
        for raw in payload["files"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                raise RustScannerError("Rust scanner returned invalid directory paths")
            paths.append(Path(raw["path"]))
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise RustScannerError("Rust scanner returned non-deterministic directory paths")
        if any(not path.resolve().is_relative_to(root_path) for path in paths):
            raise RustScannerError("Rust scanner returned a path outside the scan root")
        return self._map_batch(
            payload,
            paths,
            include_ml_candidates=include_ml_candidates,
        )

    @staticmethod
    def _directory_timeout() -> float:
        raw_timeout = os.environ.get(_DIRECTORY_TIMEOUT_ENV)
        if raw_timeout is None:
            return _DEFAULT_DIRECTORY_TIMEOUT_SECONDS
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise RustScannerError(
                f"{_DIRECTORY_TIMEOUT_ENV} must be a positive number"
            ) from exc
        if not timeout > 0:
            raise RustScannerError(
                f"{_DIRECTORY_TIMEOUT_ENV} must be a positive number"
            )
        return timeout

    def scan_file_with_ml(
        self,
        path: str | Path,
        verifier: "Verifier",
        *,
        max_bytes: Optional[int] = None,
        ml_threshold: float = 0.5,
    ) -> list[Finding]:
        """Scan a file in Rust and call the existing Python ML verifier."""
        path_obj = Path(path)
        findings = self._collect_file(path_obj, max_bytes=max_bytes)
        if not findings:
            return findings

        try:
            if max_bytes is None:
                content = path_obj.read_text(encoding="utf-8", errors="ignore")
            else:
                with path_obj.open("r", encoding="utf-8", errors="ignore") as stream:
                    content = stream.read(max_bytes)
        except OSError as exc:
            raise RustScannerError(f"failed to read {path_obj} for ML verification: {exc}") from exc
        return self._verify_candidates(findings, content, verifier, ml_threshold)

    def _collect_text(self, text: str, *, file: Optional[str]) -> list[Finding]:
        arguments = [str(self.executable), "scan-text"]
        if file is not None:
            arguments.extend(("--file", file))
        payload = self._expect_findings_payload(self._run(arguments, input_text=text))
        return self._map_findings(payload, file=file)

    def _collect_file(self, path: Path, *, max_bytes: Optional[int]) -> list[Finding]:
        arguments = [str(self.executable), "scan-file", str(path)]
        if max_bytes is not None:
            arguments.extend(("--max-bytes", str(max_bytes)))
        payload = self._expect_findings_payload(self._run(arguments))
        return self._map_findings(payload, file=str(path))

    @staticmethod
    def _run(
        arguments: list[str],
        *,
        input_text: Optional[str] = None,
        timeout: Optional[float] = 60,
    ) -> Any:
        try:
            completed = subprocess.run(
                arguments,
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RustScannerError(f"Rust scanner execution failed: {exc}") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RustScannerError(f"Rust scanner failed: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RustScannerError("Rust scanner returned invalid JSON") from exc
        return payload

    @staticmethod
    def _expect_findings_payload(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise RustScannerError("Rust scanner returned an invalid result shape")
        return payload

    @classmethod
    def _map_batch(
        cls,
        payload: Any,
        requested_paths: list[Path],
        *,
        include_ml_candidates: bool = False,
    ) -> NativeBatchResult:
        if not isinstance(payload, dict):
            raise RustScannerError("Rust scanner returned an invalid batch result shape")
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RustScannerError("Rust scanner returned an unsupported protocol version")
        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or not all(isinstance(item, dict) for item in raw_files):
            raise RustScannerError("Rust scanner returned invalid batch files")
        response_paths = [item.get("path") for item in raw_files]
        expected_paths = [str(path) for path in requested_paths]
        if response_paths != expected_paths:
            raise RustScannerError("Rust scanner returned unexpected batch file paths")

        results = []
        for raw, path in zip(raw_files, requested_paths):
            try:
                findings_payload = cls._expect_findings_payload(raw["findings"])
                scanned = raw["scanned"]
                line_count = raw["line_count"]
                bytes_scanned = raw["bytes_scanned"]
                error = raw.get("error")
                if not isinstance(scanned, bool):
                    raise TypeError("scanned must be a boolean")
                if (
                    not isinstance(line_count, int)
                    or isinstance(line_count, bool)
                    or line_count < 0
                ):
                    raise TypeError("line_count must be a non-negative integer")
                if (
                    not isinstance(bytes_scanned, int)
                    or isinstance(bytes_scanned, bool)
                    or bytes_scanned < 0
                ):
                    raise TypeError("bytes_scanned must be a non-negative integer")
                if error is not None and not isinstance(error, str):
                    raise TypeError("error must be a string or null")
                if scanned and error is not None:
                    raise ValueError("scanned files cannot contain an error")
                if not scanned and (
                    not error or findings_payload or line_count != 0 or bytes_scanned != 0
                ):
                    raise ValueError("unscanned files must contain only an error")
            except (KeyError, TypeError, ValueError) as exc:
                raise RustScannerError("Rust scanner returned invalid batch metadata") from exc
            findings = [
                finding
                for finding in cls._map_findings(findings_payload, file=str(path))
                if include_ml_candidates or finding.evidence != EvidenceType.ML
            ]
            results.append(
                NativeFileResult(
                    path=path,
                    findings=findings,
                    scanned=scanned,
                    line_count=line_count,
                    bytes_scanned=bytes_scanned,
                    error=error,
                )
            )
        return NativeBatchResult(results)

    @staticmethod
    def _map_findings(payload: list[dict[str, Any]], *, file: Optional[str]) -> list[Finding]:
        findings = []
        for index, raw in enumerate(payload):
            try:
                evidence = EvidenceType(str(raw["evidence"]))
                kind = str(raw["type"])
                token = str(raw["token"])
                var_name = raw.get("var_name")
                signature = (
                    kind if evidence == EvidenceType.REGEX and kind != "ENV_ASSIGNMENT" else None
                )
                inference = infer_category(
                    signature_name=signature,
                    var_name=str(var_name) if var_name else None,
                    token=token,
                )

                if signature is not None:
                    severity = Severity(str(raw["severity"]))
                    confidence = float(raw["confidence"])
                else:
                    severity = (
                        _severity_from_classification(inference)
                        if kind == "ENV_ASSIGNMENT"
                        else _severity_from_entropy(inference)
                    )
                    confidence = (
                        inference.confidence
                        if kind == "ENV_ASSIGNMENT"
                        else float(raw["confidence"])
                    )

                findings.append(
                    Finding(
                        type=kind,
                        file=file,
                        line=int(raw["line"]),
                        snippet=str(raw["snippet"]),
                        entropy=float(raw["entropy"]),
                        evidence=evidence,
                        severity=severity,
                        confidence=confidence,
                        token=token,
                        token_start=_optional_int(raw.get("token_start")),
                        token_end=_optional_int(raw.get("token_end")),
                        in_comment=raw.get("in_comment"),
                        category=inference.category.value,
                        category_reason=inference.reason,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RustScannerError(f"Invalid Rust finding payload at index {index}") from exc
        return findings

    @staticmethod
    def _verify_candidates(
        findings: list[Finding],
        content: str,
        verifier: "Verifier",
        ml_threshold: float,
    ) -> list[Finding]:
        regex_findings = [finding for finding in findings if finding.evidence == EvidenceType.REGEX]
        candidates = [finding for finding in findings if finding.evidence != EvidenceType.REGEX]
        if not candidates:
            return regex_findings
        verified = _apply_ml_verification(
            findings=candidates,
            full_content=content,
            verifier=verifier,
            ml_threshold=ml_threshold,
        )
        return regex_findings + verified


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expected an integer, string, or null")
    if isinstance(value, (int, str)):
        return int(value)
    raise TypeError("expected an integer, string, or null")


__all__ = [
    "NativeBatchResult",
    "NativeFileResult",
    "PROTOCOL_VERSION",
    "RustScannerBackend",
    "RustScannerError",
]
