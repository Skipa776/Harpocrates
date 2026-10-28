"""NFR-01 check: the latency the real scanner adds to a model request, against the two targets.

    python -m bench.check_latency --recordings DIR/mock [--run-prefix 20261021] [--repeats 3]
        [--max-follow-ups 10] [--out bench/results/gate]

bench.gate_scan_latency replays each recorded session (made by `python -m bench.canary_harness --mode agent --out-dir
DIR`, so --recordings is DIR/mock) directly and through a gate with the real scanner; the added latency is the
difference. Targets, on the p95: follow-up requests under 50 ms, a session's first request under 500 ms. A first
request over 1 MiB is outside its target but still counted, which can only raise the p95. Exits 1 when a target is
missed or has no sample. The numbers depend on the machine; the output holds numbers only, never a body or header.
"""

import argparse
from pathlib import Path
from typing import Optional

from bench import gate_scan_latency

TARGET_FOLLOW_UP_MS = 50
TARGET_FIRST_TURN_MS = 500
ONE_MIB = 1_048_576


def _word(passed: bool) -> str:
    """
    Description: Name the outcome of one target for the output line.
    Parameters:
        passed: whether the target was met.
    Output: "pass" or "FAIL".
    """
    return "pass" if passed else "FAIL"


def verdict(record: dict) -> tuple[list[str], bool]:
    """
    Description: Judge a record from gate_scan_latency.measure against the follow-up and first-turn targets.
    Parameters:
        record: the dict measure returns; the follow_up and first_turn groups are read, each with n, added_ms and body_bytes.
    Output: (lines, ok): one line per target, and True only when both targets were measured and met. A group with n of 0
        gives a "NOT MEASURED: ..." line and ok False. A first-turn line whose largest body is over 1 MiB ends with
        "(includes bodies over 1 MiB, outside the target)".
    """
    lines: list[str] = []
    ok = True
    follow = record.get("follow_up", {"n": 0})
    if not follow.get("n"):
        lines.append(f"NOT MEASURED: no follow-up sample, so the follow-up target (p95 under {TARGET_FOLLOW_UP_MS} ms) is not judged")
        ok = False
    else:
        p95 = follow["added_ms"]["p95"]
        passed = p95 < TARGET_FOLLOW_UP_MS
        lines.append(f"follow-up p95 added {p95} ms (target {TARGET_FOLLOW_UP_MS}, n={follow['n']}): {_word(passed)}")
        ok = ok and passed
    first = record.get("first_turn", {"n": 0})
    if not first.get("n"):
        lines.append(f"NOT MEASURED: no first-turn sample, so the first-turn target (p95 under {TARGET_FIRST_TURN_MS} ms) is not judged")
        ok = False
    else:
        p95 = first["added_ms"]["p95"]
        passed = p95 < TARGET_FIRST_TURN_MS
        largest = first["body_bytes"]["max"]
        line = (f"first-turn p95 added {p95} ms (target {TARGET_FIRST_TURN_MS}, n={first['n']}, "
                f"largest body {largest} bytes): {_word(passed)}")
        if largest > ONE_MIB:
            line += " (includes bodies over 1 MiB, outside the target)"
        lines.append(line)
        ok = ok and passed
    return lines, ok


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Measure the scanner's added latency on the recordings, save the record, and print the verdict.
    Parameters:
        argv: command-line arguments, or None to use sys.argv.
    Output: Exit status 0 when both targets are met, 1 when one is missed or not measured. Prints the path of the saved
        record, then one line per target from verdict.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, required=True, help="the recordings folder, e.g. DIR/mock from bench.canary_harness")
    parser.add_argument("--run-prefix", default="", help="only recorded runs whose folder name starts with this (default: all)")
    parser.add_argument("--repeats", type=int, default=3, help="how many times each session is replayed")
    parser.add_argument("--max-follow-ups", type=int, default=10, help="how many recorded follow-ups of a session are replayed")
    parser.add_argument("--out", type=Path, default=gate_scan_latency.DEFAULT_OUT, help="the folder for the saved record")
    args = parser.parse_args(argv)
    record = gate_scan_latency.measure(args.recordings, repeats=args.repeats, max_follow_ups=args.max_follow_ups,
                                       run_prefix=args.run_prefix)
    path = gate_scan_latency.write_record(record, args.out)
    lines, ok = verdict(record)
    print(f"record written: {path}")
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
