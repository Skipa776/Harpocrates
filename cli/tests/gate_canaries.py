"""Generated canaries for the gate tests: fake values in real provider formats, never a real credential (AGENTS.md rule 3)."""
from __future__ import annotations

import random
import re

from bench import canary_match, canary_repo
from Harpocrates.core.classification import SECRET_TYPES

PLACEHOLDER = re.compile(r"<<HARPO:(?:" + "|".join(SECRET_TYPES) + r"):[0-9a-f]{4}>>")


def canaries(seed: int = 0) -> dict[str, str]:
    """All canary kinds for a seed: {kind: value}. The same seed gives the same values."""
    random.seed(f"gate-tests:{seed}")
    return canary_repo._canaries()


def known_format(seed: int = 0) -> dict[str, str]:
    """Only the kinds a signature regex finds on its own."""
    return {kind: value for kind, value in canaries(seed).items() if kind in canary_match.KNOWN_FORMAT_KINDS}
