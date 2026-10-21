from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenMatch:
    """
    Description: Authoritative span of a detected token within its source line, carried to the features
        through CodeContext.token_match.
    Parameters:
        token: Detected candidate text.
        start: Inclusive character offset in the detection source text.
        end: Exclusive character offset in the detection source text.
        kind: Optional extraction kind, such as regex or sensitive_assignment.
    Output: An immutable record. For a regex hit the offsets count from the stripped line; for a
        sensitive-assignment candidate they count from the URL-stripped text, so they can drift from the
        line when a URL was removed.
    """

    token: str
    start: int
    end: int
    kind: Optional[str] = None  # "regex" | "sensitive_assignment"
