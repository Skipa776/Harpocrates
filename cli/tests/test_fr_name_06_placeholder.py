"""FR-NAME-06: the placeholder is `<<HARPO:{type}:{4 hex}>>` and the rename does not change it.

The format is built in Redactor.placeholder (Harpocrates/read.py) and recognised by read._PLACEHOLDER and by the hook's
placeholder count (hooks.py). The existing placeholder tests (test_read_tool.py, test_hooks.py, test_hooks_grants.py,
test_mcp_no_raw_values.py, test_path_false_positives.py) assert this format and are left as they were by the rename.
"""
from __future__ import annotations

import re

from Harpocrates import hooks, read
from Harpocrates.read import Redactor

SHAPE = re.compile(r"\A<<HARPO:([A-Za-z0-9_]+):([0-9a-f]{4})>>\Z")


def test_fr_name_06_placeholder_format_unchanged() -> None:
    redactor = Redactor()
    for stype in ("aws_key", "password", "token", "private_key"):
        text = redactor.placeholder("canary-value-123", stype)
        match = SHAPE.match(text)
        assert match, text
        assert match.group(1) == stype
        assert len(text) == len("<<HARPO::>>") + len(stype) + 4  # exactly four hex characters, nothing else
    # Stable for one value within a session, and keyed (a different session gives a different digest).
    assert redactor.placeholder("v", "t") == redactor.placeholder("v", "t")


def test_fr_name_06_the_recognisers_still_match_it() -> None:
    text = Redactor().placeholder("canary-value-123", "token")
    assert read._PLACEHOLDER.fullmatch(text)
    assert hooks._placeholders(f"a {text} b {text}") == {text: 2}
    assert hooks._PLACEHOLDER_TYPE.findall(text) == ["token"]
    assert "<<HARPO" in text and "harpocrates" not in text.lower()
    assert not read._PLACEHOLDER.fullmatch(text.replace("HARPO", "HARPOCRATES"))  # the old spelling is not one
