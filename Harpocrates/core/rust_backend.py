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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from Harpocrates.core.classification import infer_category
from Harpocrates.core.detector import (
    _apply_ml_verification,
    _severity_from_classification,
    _severity_from_entropy,
)
from Harpocrates.core.result import EvidenceType, Finding, Severity

if TYPE_CHECKING:
    from Harpocrates.ml.verifier import Verifier

_BINARY_ENV = "HARPOCRATES_RUST_SCANNER"
_BINARY_NAME = "harpocrates-rust-scanner"


class RustScannerError(RuntimeError):
    """Raised when the native scanner cannot be located or executed."""


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
                return cls(candidate)

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
        findings = self._collect_file(Path(path), max_bytes=max_bytes)
        return [finding for finding in findings if finding.evidence != EvidenceType.ML]

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
        except OSError:
            return [finding for finding in findings if finding.evidence != EvidenceType.ML]
        return self._verify_candidates(findings, content, verifier, ml_threshold)

    def _collect_text(self, text: str, *, file: Optional[str]) -> list[Finding]:
        arguments = [str(self.executable), "scan-text"]
        if file is not None:
            arguments.extend(("--file", file))
        payload = self._run(arguments, input_text=text)
        return self._map_findings(payload, file=file)

    def _collect_file(self, path: Path, *, max_bytes: Optional[int]) -> list[Finding]:
        arguments = [str(self.executable), "scan-file", str(path)]
        if max_bytes is not None:
            arguments.extend(("--max-bytes", str(max_bytes)))
        payload = self._run(arguments)
        return self._map_findings(payload, file=str(path))

    @staticmethod
    def _run(arguments: list[str], *, input_text: Optional[str] = None) -> list[dict[str, Any]]:
        try:
            completed = subprocess.run(
                arguments,
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
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
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise RustScannerError("Rust scanner returned an invalid result shape")
        return payload

    @staticmethod
    def _map_findings(payload: list[dict[str, Any]], *, file: Optional[str]) -> list[Finding]:
        findings = []
        for raw in payload:
            try:
                evidence = EvidenceType(str(raw["evidence"]))
                kind = str(raw["type"])
                token = str(raw["token"])
                var_name = raw.get("var_name")
                signature = kind if evidence == EvidenceType.REGEX and kind != "ENV_ASSIGNMENT" else None
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
                raise RustScannerError(f"Invalid Rust finding payload: {raw!r}") from exc
        return findings

    @staticmethod
    def _verify_candidates(
        findings: list[Finding],
        content: str,
        verifier: "Verifier",
        ml_threshold: float,
    ) -> list[Finding]:
        regex_findings = [
            finding for finding in findings if finding.evidence == EvidenceType.REGEX
        ]
        candidates = [
            finding for finding in findings if finding.evidence != EvidenceType.REGEX
        ]
        if not candidates:
            return regex_findings
        try:
            verified = _apply_ml_verification(
                findings=candidates,
                full_content=content,
                verifier=verifier,
                ml_threshold=ml_threshold,
            )
        except Exception:
            verified = [
                finding for finding in candidates if finding.evidence != EvidenceType.ML
            ]
        return regex_findings + verified


def _optional_int(value: object) -> Optional[int]:
    return None if value is None else int(value)


__all__ = ["RustScannerBackend", "RustScannerError"]
