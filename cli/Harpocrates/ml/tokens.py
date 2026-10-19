from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenMatch:
    """
    Description: Authoritative span of a detected token within its source line.
    Parameters:
        token: Detected candidate text.
        start: Inclusive character offset in the detection source text.
        end: Exclusive character offset in the detection source text.
        kind: Optional extraction kind, such as regex or sensitive_assignment.
    Output: Instances provide immutable token text, start/end offsets, and optional extraction kind
    with the coordinate limitations below.
    `start` and `end` are character offsets into the source text used during detection: - For
    CRITICAL/HIGH regex hits (kind="regex"): offsets are into `stripped` (the original source line
    with leading/trailing whitespace removed). These are authoritative. - For sensitive-assignment
    candidates (kind="sensitive_assignment"): offsets are into `scan_text` (URL-stripped version of
    `stripped`). URL stripping does not preserve string length, so these offsets may diverge from
    the original line when a URL was stripped. v0.3 will re-derive proper offsets from the original
    line alongside the ONNX retrain.
    Currently populated by `core/detector.py` and carried via `CodeContext.token_match`. The 8
    helpers in `ml/features.py` that call `line.find(token)` will be migrated to use these offsets
    in v0.3.
    """

    token: str
    start: int
    end: int
    kind: Optional[str] = None  # "regex" | "sensitive_assignment"
