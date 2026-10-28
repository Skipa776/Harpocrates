"""FR-PROOF-04, FR-GATE-04: the canary harness in direct mode against gate -> mock.

The corpus (94 cases, bench/gate_corpus*.py) goes through the real gate with its real scanner to the mock provider; the
recording must hold no known-format canary. ML-only canaries are counted separately. Reports hold counts, never a value.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from gate_scan_support import passthrough

from bench import canary_harness as harness
from bench import gate_corpus


@pytest.fixture(scope="module")
def gated(tmp_path_factory: pytest.TempPathFactory):
    return harness.run_direct(seed=0, out_root=tmp_path_factory.mktemp("gated"), run_id="unit-direct-gate", target="gate")


def test_fr_gate_04_direct_mode_against_the_gate_receives_no_known_format_canary(gated) -> None:
    data = gated.report
    assert data["mode"] == "direct" and data["protection"] == "gate" and data["target"] == "gate -> mock"
    # The two connector credentials (decision 6, class K) are sent on purpose and are listed apart, not counted as a leak.
    assert data["received"]["known_format"] == 0 and gated.verdict.exit_code == 0, data["verdict"]
    assert data["planted"]["by_class"]["known-format"] >= 8
    assert data["received"]["by_class"]["ml-only"] <= data["planted"]["by_class"]["ml-only"]  # reported apart, never added in


def test_fr_gate_04_direct_mode_reports_what_the_gate_did_with_each_case(gated) -> None:
    rows = {r["case_id"]: r for r in gated.report["direct"]["cases"]}
    cases = gate_corpus.build_corpus(0)
    assert set(rows) == {c.case_id for c in cases}
    for case in cases:
        row = rows[case.case_id]
        assert row["gate_status"] in (200, 400), case.case_id
        if case.canary_class == "known-format" and case.expect != "pass":
            assert not row["received"], case.case_id
        if case.case_id == "anthropic_unknown_block_blocked":
            assert row["gate_status"] == 400 and not row["received"]
        if case.case_id == "anthropic_mcp_servers_authorization_token":  # a connector credential: class K, passes on purpose
            assert row["gate_status"] == 200 and row["received"] and row["by_design"]
    summary = gated.report["gate"]
    assert summary["forwarded"] + summary["blocked"] == len(cases) and summary["blocked"] >= 10
    assert gated.report["requests"]["total"] == summary["forwarded"]  # a blocked case never reached the mock
    assert summary["received_by_design"] == ["anthropic_mcp_servers_authorization_token", "responses_tools_mcp_authorization"]


def test_fr_gate_04_direct_mode_notices_a_gate_that_lets_a_known_format_canary_through(tmp_path: Path) -> None:
    leaky = harness.run_direct(seed=0, out_root=tmp_path, run_id="unit-direct-leak", target="gate", gate_scan_step=passthrough)
    assert leaky.verdict.exit_code == 1 and "known-format" in leaky.verdict.reason
    assert leaky.report["received"]["known_format"] >= 8


def test_fr_gate_04_direct_mode_against_the_gate_writes_no_value(gated) -> None:
    text = gated.report_path.read_text()
    assert all(c.canary_value not in text for c in gate_corpus.build_corpus(0) if c.canary_value)
    assert (gated.report_path.parent.parent / "mock" / "unit-direct-gate" / "requests.jsonl").exists()


def test_fr_gate_04_the_cli_takes_target_gate_for_direct_mode_only(tmp_path: Path, capsys) -> None:
    code = harness.main(["--mode", "direct", "--target", "gate", "--seed", "1", "--out-dir", str(tmp_path), "--run-id", "cli-gate"])
    out = capsys.readouterr().out
    assert code == 0 and "target=gate -> mock" in out and json.loads((tmp_path / "proof" / "cli-gate.json").read_text())
    with pytest.raises(SystemExit) as stop:
        harness.main(["--mode", "agent", "--target", "gate", "--out-dir", str(tmp_path)])
    assert stop.value.code == 2  # agent mode through the gate is bench/gate_e2e.py
