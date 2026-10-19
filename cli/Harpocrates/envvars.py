"""Environment variables (FR-NAME-04): HARPO_<NAME> first, then the old HARPOCRATES_<NAME>.

getenv() is the one place that knows both prefixes. The variables read today are RUST_SCANNER, RUST_TIMEOUT_SECONDS
and REQUIRE_RUST (the last by the Rust parity test and CI); the Rust core reads RUST_WORKERS the same way.

There is no DIR variable, on purpose: the state directory (paths.state_dir) cannot be moved by the environment,
because a project's settings can set environment variables and a state directory the agent can write would replace
the allowlist. Neither HARPO_DIR nor HARPOCRATES_DIR is read.
"""
from __future__ import annotations

import os
from typing import Optional

PREFIXES = ("HARPO_", "HARPOCRATES_")  # in the order they are tried


def getenv(suffix: str) -> Optional[str]:
    """
    Description: The value of HARPO_<suffix>, or HARPOCRATES_<suffix> when that is not set.
    Parameters:
        suffix: the variable name without its prefix, for example "RUST_SCANNER".
    Output: The value, or None when neither is set. A variable that is set but empty counts as set, so an empty
        HARPO_ value wins over a HARPOCRATES_ one. No side effects.
    """
    for prefix in PREFIXES:
        value = os.environ.get(prefix + suffix)
        if value is not None:
            return value
    return None
