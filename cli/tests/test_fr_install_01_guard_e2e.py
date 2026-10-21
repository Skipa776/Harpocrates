"""FR-INSTALL-01, FR-INSTALL-02, FR-INSTALL-08 and FR-UX-01: the verdict of bench/guard_e2e.py, and a run with real agents.

The verdict tests judge hand-made reports and need no agent. The integration test runs `claude -p` and `codex exec`
in throwaway homes, so it needs at least one of them on PATH.
"""
from __future__ import annotations

import copy
import json
import shutil

import pytest

from bench import guard_e2e as g7


def _good_report() -> dict:
    """A report in which every scenario did what its mode requires."""
    routed = {
        "scenario": "claude-guard-routed", "agent": "claude", "mode": "routed", "status": "ok",
        "guard_up_exit": 0, "routing_written": True, "tool_results_seen": 2,
        "doctor_exit": 0, "received": {"known_format": 0},
        "guard_down_exit": 0, "restored_exactly": True,
    }
    after = {
        "scenario": "codex-guard-after-down", "agent": "codex", "mode": "after-down", "status": "ok",
        "guard_up_exit": 0, "routing_written": True, "tool_results_seen": 2,
        "received": {"known_format": 3},
        "guard_down_exit": 0, "restored_exactly": True,
    }
    return {"isolation": {"real_home_changed": []}, "scenarios": [routed, after]}


def test_fr_install_01_guard_e2e_has_routed_and_after_down_for_both_agents():
    combos = {(s.agent, s.mode) for s in g7.SCENARIOS}
    assert combos == {("claude", "routed"), ("claude", "after-down"), ("codex", "routed"), ("codex", "after-down")}
    assert len(g7.SCENARIOS) == 4
    labels = [s.label for s in g7.SCENARIOS]
    assert len(set(labels)) == len(labels), labels


def test_fr_install_01_guard_e2e_verdict_accepts_a_good_report():
    assert g7.passed(_good_report()) is True


@pytest.mark.parametrize(("where", "path", "value"), [
    pytest.param("routed", ("received", "known_format"), 1, id="routed-known-format-leaks"),
    pytest.param("routed", ("doctor_exit",), 1, id="routed-doctor-fails"),
    pytest.param("routed", ("restored_exactly",), False, id="routed-not-restored"),
    pytest.param("after", ("received", "known_format"), 0, id="after-down-control-did-not-leak"),
    pytest.param("after", ("tool_results_seen",), 0, id="after-down-no-tool-results"),
    pytest.param("isolation", ("real_home_changed",), ["~/.claude/settings.json"], id="real-home-changed"),
])
def test_fr_install_01_guard_e2e_verdict_rejects_each_failure(where, path, value):
    assert g7.passed(_good_report())  # the base report passes, so each mutation below is what fails it
    data = copy.deepcopy(_good_report())
    node = {"routed": data["scenarios"][0], "after": data["scenarios"][1], "isolation": data["isolation"]}[where]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert g7.passed(data) is False


def test_fr_install_01_guard_e2e_verdict_ignores_skipped_scenarios():
    data = {
        "isolation": {"real_home_changed": []},
        "scenarios": [{"scenario": "codex-guard-routed", "agent": "codex", "mode": "routed", "status": "skipped"}],
    }
    assert g7.passed(data) is True


@pytest.mark.integration
def test_fr_install_01_guard_e2e_real_agents_routed_by_guard_up(tmp_path):
    if shutil.which("claude") is None and shutil.which("codex") is None:
        pytest.skip("neither claude nor codex is on PATH")
    data, path = g7.run_suite(0, tmp_path, "test-guard-e2e")
    assert path.is_file() and g7.passed(data), json.dumps(
        [{k: v for k, v in s.items() if k != "planted"} for s in data["scenarios"]])[:2000]
