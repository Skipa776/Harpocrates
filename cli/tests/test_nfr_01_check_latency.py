"""NFR-01: the pass/fail judgement of the scanner-latency check, on hand-written records (no timing)."""

from bench import check_latency, gate_scan_latency


def _part(n, p95, max_bytes=1_000):
    return {"n": n, "body_bytes": {"min": 1, "p50": 1, "max": max_bytes},
            "added_ms": {"p50": p95, "p95": p95, "mean": p95, "min": p95, "max": p95}}


def _record(follow, first):
    return {"follow_up": follow, "first_turn": first}


def test_nfr_01_all_targets_under_pass():
    lines, ok = check_latency.verdict(_record(_part(40, 12.5), _part(8, 300.0)))
    assert ok
    assert len(lines) == 2
    assert all(line.endswith(": pass") for line in lines)


def test_nfr_01_follow_up_over_50_fails():
    lines, ok = check_latency.verdict(_record(_part(40, 51.0), _part(8, 300.0)))
    assert not ok
    assert lines[0].endswith(": FAIL")
    assert lines[1].endswith(": pass")


def test_nfr_01_follow_up_at_50_fails():
    _, ok = check_latency.verdict(_record(_part(40, 50.0), _part(8, 300.0)))
    assert not ok


def test_nfr_01_first_turn_over_500_fails():
    lines, ok = check_latency.verdict(_record(_part(40, 10.0), _part(8, 501.0)))
    assert not ok
    assert lines[0].endswith(": pass")
    assert lines[1].endswith(": FAIL")


def test_nfr_01_first_turn_at_500_fails():
    _, ok = check_latency.verdict(_record(_part(40, 10.0), _part(8, 500.0)))
    assert not ok


def test_nfr_01_no_follow_ups_not_measured():
    lines, ok = check_latency.verdict(_record({"n": 0}, _part(8, 300.0)))
    assert not ok
    assert lines[0].startswith("NOT MEASURED")
    assert lines[1].endswith(": pass")


def test_nfr_01_no_first_turn_not_measured():
    lines, ok = check_latency.verdict(_record(_part(40, 10.0), {"n": 0}))
    assert not ok
    assert lines[0].endswith(": pass")
    assert lines[1].startswith("NOT MEASURED")


def test_nfr_01_first_turn_body_over_1_mib_is_judged_and_marked():
    lines, ok = check_latency.verdict(_record(_part(40, 10.0), _part(8, 300.0, max_bytes=2_000_000)))
    assert ok
    assert "largest body 2000000 bytes" in lines[1]
    assert lines[1].endswith("(includes bodies over 1 MiB, outside the target)")


def test_nfr_01_first_turn_body_of_exactly_1_mib_is_not_marked():
    lines, _ = check_latency.verdict(_record(_part(40, 10.0), _part(8, 300.0, max_bytes=1_048_576)))
    assert "outside the target" not in lines[1]


def test_nfr_01_main_exit_status_follows_verdict(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_write(record, out_dir):
        return out_dir / "scan-latency-test.json"

    def fake_measure(recordings, repeats, max_follow_ups, run_prefix):
        seen.update(repeats=repeats, max_follow_ups=max_follow_ups, run_prefix=run_prefix)
        return fake_record

    monkeypatch.setattr(gate_scan_latency, "write_record", fake_write)
    monkeypatch.setattr(gate_scan_latency, "measure", fake_measure)
    args = ["--recordings", str(tmp_path), "--out", str(tmp_path)]

    fake_record = _record(_part(40, 12.0), _part(8, 300.0))
    assert check_latency.main(args) == 0
    assert seen == {"repeats": 3, "max_follow_ups": 10, "run_prefix": ""}

    fake_record = _record({"n": 0}, _part(8, 300.0))
    assert check_latency.main(args) == 1
    assert "NOT MEASURED" in capsys.readouterr().out
