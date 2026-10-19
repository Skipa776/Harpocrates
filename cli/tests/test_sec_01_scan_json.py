"""SEC-01: `harpo scan --json` never carries a plaintext secret unless --show-secrets is given.

Every secret here is a generated canary (bench/canary_repo.py or built at runtime); none is a real credential.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench.canary_repo import build
from Harpocrates.cli import app
from Harpocrates.core.result import EvidenceType, Finding, ScanResult
from Harpocrates.utils.redaction import mask_secrets, redact_token

runner = CliRunner()
_RUN = 12  # a run this long of a secret in the output counts as a leak (same bar as test_mcp_no_raw_values)


@pytest.fixture(scope="module")
def canary_repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    repo = tmp_path_factory.mktemp("sec01") / "repo"
    return repo, build(repo, seed=0)


def _secret_lines(value: str) -> list[str]:
    """The lines of a canary that are secret (a PEM's BEGIN/END lines are public labels)."""
    return [ln for ln in value.splitlines() if ln and not ln.startswith("-----")]


def _scan_json(repo: Path, *flags: str) -> tuple[str, dict]:
    result = runner.invoke(app, ["scan", str(repo), "--json", *flags])
    assert result.exit_code in (0, 1), result.output
    return result.stdout, json.loads(result.stdout)


def _leaks(out: str, manifest: list[dict]) -> list[str]:
    found = []
    for entry in manifest:
        for line in _secret_lines(entry["value"]):
            if line in out:
                found.append(f"{entry['kind']}: whole line")
            for i in range(max(len(line) - _RUN + 1, 0)):
                if line[i : i + _RUN] in out:
                    found.append(f"{entry['kind']}: {_RUN}-char run at {i}")
                    break
    return found


def test_sec_01_scan_json_redacts_snippet(canary_repo: tuple[Path, list[dict]]) -> None:
    repo, manifest = canary_repo
    out, data = _scan_json(repo)
    assert data["findings"]
    assert _leaks(out, manifest) == []
    # The line stays readable: the secret is swapped for the form the table shows.
    github = next(e["value"] for e in manifest if e["kind"] == "github_token")
    snippets = [f["snippet"] for f in data["findings"] if f["file"].endswith("settings.py")]
    assert f'GITHUB_TOKEN = "{redact_token(github)}"' in snippets
    stripe = next(e["value"] for e in manifest if e["kind"] == "stripe_key")
    assert f'stripe_secret: "{redact_token(stripe)}"' in [f["snippet"] for f in data["findings"]]


def test_sec_01_scan_json_show_secrets_still_full(canary_repo: tuple[Path, list[dict]]) -> None:
    repo, manifest = canary_repo
    out, data = _scan_json(repo, "--show-secrets")
    github = next(e["value"] for e in manifest if e["kind"] == "github_token")
    assert f'GITHUB_TOKEN = "{github}"' in [f["snippet"] for f in data["findings"]]
    assert github in out


def test_sec_01_scan_json_default_matches_help(canary_repo: tuple[Path, list[dict]]) -> None:
    """The documented default: redacted in JSON; only --show-secrets opts in."""
    repo, _ = canary_repo
    help_text = " ".join(runner.invoke(app, ["scan", "--help"]).stdout.split())
    assert "redacted in both table and JSON output" in help_text
    _, redacted = _scan_json(repo)
    _, full = _scan_json(repo, "--show-secrets")
    assert [f["file"] for f in redacted["findings"]] == [f["file"] for f in full["findings"]]
    assert {k for f in redacted["findings"] for k in f} | {"token"} == {k for f in full["findings"] for k in f}


def test_sec_01_scan_json_explain_redacts_snippet(canary_repo: tuple[Path, list[dict]]) -> None:
    pytest.importorskip("xgboost", reason="--explain needs xgboost")
    repo, manifest = canary_repo
    result = runner.invoke(app, ["scan", str(repo), "--json", "--explain"])
    assert result.exit_code in (0, 1), result.output
    assert json.loads(result.stdout)["findings"]
    assert _leaks(result.stdout, manifest) == []


def _finding(snippet: str, token: str, line: int = 1) -> Finding:
    return Finding(type="T", snippet=snippet, evidence=EvidenceType.REGEX, file="f.env", line=line, token=token,
                   category_reason=f"layer=x matched=/{token}/")


def test_sec_01_short_token_is_fully_masked_in_snippet() -> None:
    """redact_token shows 8 of 11 characters; in a snippet a token under 16 is masked whole."""
    for token in ("abcd1234xyz", "abcdefgh", "abcdefghij", "ab12"):
        out = ScanResult(findings=[_finding(f"PW={token}", token)]).to_dict()["findings"][0]
        assert out["snippet"] == "PW=[REDACTED]", token
        assert token not in json.dumps(out)
    token = "Aa1" * 6  # 18 characters: the table form
    out = ScanResult(findings=[_finding(f"PW={token}", token)]).to_dict()["findings"][0]
    assert out["snippet"] == f"PW={redact_token(token)}"


def test_sec_01_every_occurrence_overlap_and_sibling_tokens_are_masked() -> None:
    a, b = "AKIA" + "Q7" * 8, "ghp_" + "Zk9" * 8
    line = f'x="{a}" y="{b}" z="{a}"'
    # Two findings on one line: each snippet masks both secrets, and both occurrences of the first.
    result = ScanResult(findings=[_finding(line, a), _finding(line, b)]).to_dict()
    for f in result["findings"]:
        assert a not in f["snippet"] and b not in f["snippet"]
        assert f["snippet"].count(redact_token(a)) == 2
    # Partly overlapping tokens leave nothing of either in the output.
    left, right = "0123456789abcdefghij", "abcdefghijKLMNOPQRST"
    masked = mask_secrets(f"v={left}KLMNOPQRST!", [left, right])
    assert "abcdefghij" not in masked and "KLMNO" not in masked and masked.endswith("!")
    # Findings on other lines do not mask this line.
    other = ScanResult(findings=[_finding(line, a, line=1), _finding("k=1", b, line=2)]).to_dict()["findings"]
    assert b in other[0]["snippet"]


def test_sec_01_snippet_cut_inside_the_token_is_still_masked() -> None:
    token = "eyJ" + "Qw8Zr1" * 30
    snippet = ("x = '" + token)[:100]  # the detector keeps the first 200 characters of the line
    assert len(snippet) < len("x = '" + token) and token not in snippet
    out = ScanResult(findings=[_finding(snippet, token)]).to_dict()["findings"][0]["snippet"]
    assert "Qw8Zr1Qw8Zr1" not in out and out.startswith("x = '")


def test_sec_01_category_reason_does_not_carry_the_token() -> None:
    token = "sk_live_" + "Tt5Yy7" * 4
    out = ScanResult(findings=[_finding(f"k={token}", token)]).to_dict()["findings"][0]
    assert token not in out["category_reason"]
    shown = ScanResult(findings=[_finding(f"k={token}", token)]).to_dict(include_token=True)["findings"][0]
    assert shown["snippet"] == f"k={token}" and shown["token"] == token
