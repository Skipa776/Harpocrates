"""FR-PROOF-04: counting canaries in a recording, the exit verdict, and a report that holds no value.

The recordings here are made up; the canaries come from the generators.
"""
from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import pytest

from bench import canary_repo
from bench import canary_report as report


def canaries(agent: str | None = None) -> list[report.Canary]:
    random.seed(11)
    values = canary_repo._canaries()
    return [report.Canary(f"{agent or 'x'}:{kind}", kind, value, agent) for kind, value in values.items()]


def record(index: int, body: object, route: str = "anthropic.messages") -> dict:
    return {"index": index, "route": route, "body": json.dumps(body), "body_status": "ok"}


def test_fr_proof_04_report_canary_repr_and_str_hide_the_value() -> None:
    first = canaries()[0]
    assert first.value not in repr(first) and first.value not in str(first)
    assert first.cls == "known-format" and first.kind == "aws_access_key_id"


def test_fr_proof_04_report_finds_each_received_canary_with_its_place() -> None:
    cs = canaries("claude")
    by_kind = {c.kind: c for c in cs}
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": f"T={by_kind['github_token'].value}"}]}]}]}
    other = {"system": f"pw {by_kind['postgres_password'].value}"}
    findings = report.find_received([record(0, body), record(1, other), {"index": 2, "route": "unknown", "body": None}], cs)
    assert [(f.request, f.kind, f.cls, f.path, f.block_type, f.form, f.agent) for f in findings] == [
        (0, "github_token", "known-format", "messages[0].content[0].content[0].text", "text", "plain", "claude"),
        (1, "postgres_password", "ml-only", "system", "system", "plain", "claude")]


def test_fr_proof_04_report_counts_planted_and_received_by_kind_and_class() -> None:
    cs = canaries("codex")
    leaked = [c for c in cs if c.kind in ("github_token", "db_password")]
    findings = report.find_received([record(0, {"input": [c.value for c in leaked] + [leaked[0].value]})], cs)
    planted = report.counts(cs)
    got = report.received_counts(cs, findings)
    assert planted["total"] == 15 and planted["by_class"] == {"known-format": 11, "ml-only": 4}
    assert planted["by_kind"]["github_token"] == 1
    assert got == {"total": 2, "known_format": 1, "by_kind": {"db_password": 1, "github_token": 1},
                   "by_class": {"known-format": 1, "ml-only": 1}}  # a canary seen twice still counts once


def test_fr_proof_04_report_counts_per_agent() -> None:
    cs = canaries("claude") + canaries("codex")
    findings = report.find_received([record(0, {"a": cs[0].value})], cs)
    assert report.received_counts([c for c in cs if c.agent == "claude"], findings)["total"] == 1
    assert report.received_counts([c for c in cs if c.agent == "codex"], findings)["total"] == 1  # same values: both


@pytest.mark.parametrize("mode, protection, known, ran, expected", [
    ("direct", "none", 0, True, (0, "ok")),
    ("agent", "read-tool", 0, True, (0, "ok")),
    ("agent", "read-tool", 2, True, (1, "known-format")),
    ("agent", "gate", 1, True, (1, "known-format")),
    ("agent", "none", 3, True, (0, "ok")),  # no protection claimed: leaks are reported, not failures
    ("agent", "read-tool", 0, False, (2, "inconclusive")),
    ("agent", "read-tool", 2, False, (1, "known-format")),  # a leak is a failure even when another agent was skipped
])
def test_fr_proof_04_report_verdict_exit_codes(mode: str, protection: str, known: int, ran: bool, expected: tuple) -> None:
    got = report.verdict(mode, protection, known, all_ran=ran)
    assert got.exit_code == expected[0] and expected[1] in got.reason


def test_fr_proof_04_report_direct_mode_fails_when_a_planted_canary_is_not_received() -> None:
    assert report.direct_verdict(planted=10, received=10).exit_code == 0
    bad = report.direct_verdict(planted=10, received=9)
    assert bad.exit_code == 1 and "9 of 10" in bad.reason


def test_fr_proof_04_report_never_writes_a_value(tmp_path: Path) -> None:
    cs = canaries("claude")
    clean = {"run_id": "r", "planted": report.counts(cs), "note": "counts only"}
    path = tmp_path / "proof" / "r.json"
    report.write_report(path, clean, cs)
    assert json.loads(path.read_text())["run_id"] == "r"
    for c in cs:
        assert c.value not in path.read_text()
    for leak in (cs[0].value, cs[2].value.encode().hex(), base64.b64encode(cs[3].value.encode()).decode()):
        dirty = {"run_id": "r", "oops": f"has {leak}"}
        with pytest.raises(report.ReportLeakError):
            report.write_report(tmp_path / "proof" / "dirty.json", dirty, cs)
        assert not (tmp_path / "proof" / "dirty.json").exists()  # nothing is written


def test_fr_proof_04_report_leak_check_covers_one_line_of_a_multiline_value(tmp_path: Path) -> None:
    pem = next(c for c in canaries() if c.kind == "private_key")
    line = pem.value.splitlines()[2]
    with pytest.raises(report.ReportLeakError):
        report.write_report(tmp_path / "r.json", {"x": line}, [pem])


def test_fr_proof_04_report_a_canary_too_short_to_be_counted_is_refused_not_silently_missed() -> None:
    with pytest.raises(ValueError, match="too short"):
        report.Canary("x:short", "postgres_password", "abc12")
    assert report.Canary("x:ok", "postgres_password", "Admin-").cls == "ml-only"
