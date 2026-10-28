"""FR-PROOF-04: the canary harness in direct mode, against the bare mock provider.

Direct mode sends the synthetic corpus (bench/gate_corpus.py) to a mock provider on 127.0.0.1 and counts, in the
recording, the canaries that arrived. Against the bare mock every planted canary must arrive: that is the proof that the
counter works. Reports hold counts and field paths, never a value. All canaries are generated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import canary_harness as harness
from bench import canary_match as match
from bench import gate_corpus


def test_fr_proof_04_harness_direct_counts_every_block_type(tmp_path: Path) -> None:
    result = harness.run_direct(seed=0, out_root=tmp_path, run_id="unit-direct")
    cases = gate_corpus.build_corpus(0)
    planted = [c for c in cases if c.canary_value]
    data = result.report
    assert data["mode"] == "direct" and data["protection"] == "none" and data["target"] == "mock"
    assert data["planted"]["total"] == len(planted) and data["received"]["total"] == len(planted)
    assert result.verdict.exit_code == 0 and data["verdict"] == {"exit_code": 0, "reason": "ok"}
    assert data["planted"]["by_class"]["known-format"] and data["planted"]["by_class"]["ml-only"]
    assert data["received"]["by_class"] == data["planted"]["by_class"]
    rows = {row["case_id"]: row for row in data["direct"]["cases"]}
    assert set(rows) == {c.case_id for c in cases}  # every block, item and tool type of the corpus was sent
    for case in cases:
        row = rows[case.case_id]
        assert row["received"] == (case.canary_value is not None), case.case_id
        if case.canary_value:
            assert row["found_paths"] == [case.field_path], case.case_id  # found exactly where it was planted
    assert data["requests"]["total"] == len(cases) and data["requests"]["by_route"] == {
        "anthropic.messages": sum(c.fmt == "anthropic" for c in cases),
        "openai.responses": sum(c.fmt == "responses" for c in cases)}
    # The recording is kept under the run's folder in the (gitignored) mock results area.
    lines = (tmp_path / "mock" / "unit-direct" / "requests.jsonl").read_text().splitlines()
    assert len(lines) == len(cases)
    assert result.report_path == tmp_path / "proof" / "unit-direct.json" and result.report_path.exists()


def test_fr_proof_04_harness_direct_notices_a_canary_that_did_not_arrive(tmp_path: Path, monkeypatch) -> None:
    real = harness.post_case

    def strip_one(url: str, case):
        if case.case_id == "anthropic_text":
            body = json.loads(json.dumps(case.body))
            body["messages"][0]["content"][0]["text"] = "<<HARPO:github_token:ab12>> removed in flight"
            return real(url, case, body)
        return real(url, case)

    monkeypatch.setattr(harness, "post_case", strip_one)
    result = harness.run_direct(seed=0, out_root=tmp_path, run_id="unit-missing")
    assert result.verdict.exit_code == 1 and "counter check failed" in result.verdict.reason
    missed = [r["case_id"] for r in result.report["direct"]["cases"] if r["canary_kind"] and not r["received"]]
    assert missed == ["anthropic_text"]


def test_fr_proof_04_harness_reports_without_values(tmp_path: Path) -> None:
    result = harness.run_direct(seed=3, out_root=tmp_path, run_id="unit-values")
    text = result.report_path.read_text()
    for case in gate_corpus.build_corpus(3):
        if case.canary_value:
            assert case.canary_value not in text
            for form, needle in match.needles(case.canary_value):
                assert needle not in text, (case.case_id, form)
    data = json.loads(text)
    assert all("value" not in row for row in data["direct"]["cases"])
    assert "canary_value" not in text
    # The recording is not the report: it keeps request bodies, so it lives in the ignored folder only.
    assert (tmp_path / "mock" / "unit-values" / "requests.jsonl").exists()


def test_fr_proof_04_harness_run_ids_name_the_mode_and_seed() -> None:
    run = harness.new_run_id("direct", 7)
    assert run.endswith("-direct-s7") and len(run.split("-")[0]) == 16 and run.split("-")[0].endswith("Z")


def test_fr_proof_04_harness_cli_writes_the_report_and_exits_with_the_verdict(tmp_path: Path, capsys) -> None:
    code = harness.main(["--mode", "direct", "--seed", "1", "--out-dir", str(tmp_path), "--run-id", "cli-run"])
    out = capsys.readouterr().out
    assert code == 0 and "cli-run" in out and (tmp_path / "proof" / "cli-run.json").exists()
    printed = [c for c in gate_corpus.build_corpus(1) if c.canary_value]
    assert not any(c.canary_value in out for c in printed)  # the console shows counts only
    assert "planted" in out and "received" in out


def test_fr_proof_04_harness_cli_refuses_a_target_that_does_not_exist(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        harness.main(["--mode", "direct", "--target", "nowhere", "--out-dir", str(tmp_path)])
    assert stop.value.code == 2
    assert not list(tmp_path.glob("**/*.json"))


def test_fr_proof_04_harness_never_proxies_the_mock_traffic(monkeypatch, tmp_path: Path) -> None:
    """A proxy variable in the environment must not carry canary-laden requests anywhere but the mock."""
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    result = harness.run_direct(seed=0, out_root=tmp_path, run_id="unit-proxy")
    assert result.verdict.exit_code == 0
