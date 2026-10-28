"""FR-PROOF-03: `harpo proof` and the doctor's "proof log" check, run in-process."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from cli_entry import CommandNameRunner

from Harpocrates import audit
from Harpocrates.cli import app

runner = CommandNameRunner()


def _state() -> Path:
    """The state folder of this test's made-up home (~/.harpo)."""
    return Path(os.environ["HOME"]) / ".harpo"


@pytest.fixture
def state() -> Path:
    """A state folder holding a five-entry proof log, each entry with one api_token placeholder."""
    folder = _state()
    folder.mkdir(parents=True, exist_ok=True)
    chain = audit.ProofChain(folder)
    for _ in range(5):
        chain.append("anthropic", "anthropic.messages", b"x", b"<<HARPO:api_token:abcd>>")
    return folder


def _tamper(state: Path) -> None:
    """Change the bytes field of the third entry in place, so its hash no longer matches."""
    path = state / audit.PROOF
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[2])
    entry["bytes"] += 1
    lines[2] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _proof_log_check(result: object) -> dict:
    """The one "proof log" entry in the doctor's JSON output."""
    checks = json.loads(result.output)  # type: ignore[attr-defined]
    found = [c for c in checks if c["name"] == "proof log"]
    assert len(found) == 1, f"expected one proof log check, got {len(found)}"
    return found[0]


def test_fr_proof_03_last_shows_newest_entries(state: Path) -> None:
    """`proof --last 2` prints the entries for seq 4 and 5, their counts and the verified line."""
    result = runner.invoke(app, ["proof", "--last", "2"])
    assert result.exit_code == 0, result.output
    seqs = [int(m.group(1)) for m in re.finditer(r"^ +(\d+) ", result.output, re.M)]
    assert seqs == [4, 5]
    assert "api_token 1" in result.output
    assert result.output.splitlines()[-1] == "Chain verifies: 5 entries."


def test_fr_proof_03_json_lists_entries_without_body(state: Path) -> None:
    """`proof --json` reports the five entries, the verified result, and no request body in any entry."""
    result = runner.invoke(app, ["proof", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data["entries"]) == 5
    assert data["total"] == 5
    assert data["verified"] is True
    assert data["problem"] is None
    for entry in data["entries"]:
        assert "sha256" in entry and "placeholders" in entry
        assert entry["placeholders"] == {"api_token": 1}
        assert "body" not in entry


def test_fr_proof_03_tampered_entry_does_not_verify(state: Path) -> None:
    """A changed entry makes `proof` exit 1 with "Chain does not verify", and --json says verified false."""
    _tamper(state)
    result = runner.invoke(app, ["proof"])
    assert result.exit_code == 1
    assert "Chain does not verify" in result.output
    result = runner.invoke(app, ["proof", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["verified"] is False
    assert isinstance(data["problem"], str)


@pytest.mark.parametrize("make_folder", [False, True], ids=["no-state-folder", "empty-state-folder"])
def test_fr_proof_03_empty_state_verifies_zero_entries(make_folder: bool) -> None:
    """With no proof log yet, `proof` exits 0 and says the chain verifies with 0 entries."""
    if make_folder:
        _state().mkdir(parents=True, exist_ok=True)
    result = runner.invoke(app, ["proof"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "Chain verifies: 0 entries."


def test_fr_proof_03_symlinked_state_folder_exits_1(tmp_path: Path) -> None:
    """A state folder that is a symlink is refused: `proof` exits 1 and names the problem."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    _state().symlink_to(target, target_is_directory=True)
    result = runner.invoke(app, ["proof"])
    assert result.exit_code == 1
    assert "symlink" in result.output


def test_fr_proof_03_doctor_proof_log_check(state: Path) -> None:
    """The doctor's "proof log" check is ok for an intact chain and fail after a tampered entry."""
    result = runner.invoke(app, ["doctor", "--json"])
    assert _proof_log_check(result)["status"] == "ok"
    _tamper(state)
    result = runner.invoke(app, ["doctor", "--json"])
    assert _proof_log_check(result)["status"] == "fail"
