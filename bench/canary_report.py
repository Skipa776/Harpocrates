"""Count the canaries a mock provider received, decide the exit status, and write a report with no value in it
(FR-PROOF-04).

A Canary is a planted value with its kind. find_received matches every canary against every recorded request body
in memory (bench/canary_match.py) and returns where each was found. The report holds counts, kinds, classes and
field paths. write_report refuses to write a report in which any value, or any encoded form of one, appears.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from bench import canary_match as match


class ReportLeakError(Exception):
    """
    Description: A report that was about to be written contains a canary value.
    Parameters:
        None: the message names the canary's id and form, never the value.
    Output: Raised by write_report before anything is written.
    """


@dataclass(frozen=True)
class Canary:
    """
    Description: One planted canary.
    Parameters:
        ident: a unique id that is safe to print, such as claude:.env:github_token or case:anthropic_text.
        kind: a key of bench/canary_repo.py's canaries.
        value: the planted value; held in memory only and kept out of repr and str.
        agent: the agent session that held it (claude, codex), or None.
    Output: An immutable record; cls gives known-format or ml-only.
    """

    ident: str
    kind: str
    value: str = field(repr=False)
    agent: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Description: Refuse a canary too short for the matcher to count, so a leak of it can never go unseen.
        Parameters: None
        Output: None. Raises ValueError naming the canary's ident (never its value) when it is under canary_match.MIN_LEN.
        """
        if len(self.value) < match.MIN_LEN:
            raise ValueError(f"canary {self.ident} is too short to be counted (under {match.MIN_LEN} characters)")

    @property
    def cls(self) -> str:
        """
        Description: The canary's class.
        Parameters: None
        Output: "known-format" or "ml-only" (ValueError for an unknown kind).
        """
        return match.kind_class(self.kind)

    def __str__(self) -> str:
        """
        Description: Print the canary without its value.
        Parameters: None
        Output: The ident and kind.
        """
        return f"Canary({self.ident}, {self.kind})"


class Finding(NamedTuple):
    """
    Description: One place a planted canary was found in a recorded request.
    Parameters:
        request: the recording's request index.
        route: the route name the request was classified as.
        ident: the canary's ident.
        kind: the canary's kind.
        cls: known-format or ml-only.
        path: the field path.
        block_type: the enclosing block or item type.
        form: plain, line, base64, hex or urlencoded.
        agent: the agent session, or None.
    Output: An immutable record holding no value.
    """

    request: int
    route: str
    ident: str
    kind: str
    cls: str
    path: str
    block_type: str
    form: str
    agent: Optional[str]


class Verdict(NamedTuple):
    """
    Description: The outcome of a run.
    Parameters:
        exit_code: 0 passed, 1 a canary reached the provider through a protected path, 2 inconclusive.
        reason: a short explanation without values.
    Output: An immutable pair.
    """

    exit_code: int
    reason: str


def find_received(records: Iterable[dict], canaries: Iterable[Canary]) -> list[Finding]:
    """
    Description: Match every canary against every recorded request body.
    Parameters:
        records: recording entries (dicts with index, route and body); an entry without a text body is skipped.
        canaries: the planted canaries.
    Output: One Finding per place and form a canary was found, in request order. Values are matched in memory only.
    """
    canary_list = list(canaries)
    findings: list[Finding] = []
    for rec in records:
        body = rec.get("body")
        if not isinstance(body, str):
            continue
        for canary in canary_list:
            for hit in match.locate(body, canary.value):
                findings.append(Finding(rec.get("index", -1), rec.get("route", ""), canary.ident, canary.kind, canary.cls,
                                        hit.path, hit.block_type, hit.form, canary.agent))
    return findings


def counts(canaries: Iterable[Canary]) -> dict:
    """
    Description: Count canaries by kind and class.
    Parameters:
        canaries: the canaries to count.
    Output: {"total": n, "by_kind": {kind: n}, "by_class": {"known-format": n, "ml-only": n}}, keys sorted.
    """
    items = list(canaries)
    by_kind: dict[str, int] = {}
    by_class = {"known-format": 0, "ml-only": 0}
    for c in items:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        by_class[c.cls] += 1
    return {"total": len(items), "by_kind": dict(sorted(by_kind.items())), "by_class": by_class}


def received_counts(canaries: Iterable[Canary], findings: Iterable[Finding]) -> dict:
    """
    Description: Count the distinct canaries that were found at least once.
    Parameters:
        canaries: the canaries in scope (one agent's, or all).
        findings: the findings from find_received.
    Output: {"total": n, "known_format": n, "by_kind": {...}, "by_class": {...}} for the canaries in scope with a finding;
        a canary seen in several requests counts once. A finding matches a canary by ident.
    """
    seen = {f.ident for f in findings}
    got = counts(c for c in canaries if c.ident in seen)
    return {"total": got["total"], "known_format": got["by_class"]["known-format"], "by_kind": got["by_kind"],
            "by_class": got["by_class"]}


def verdict(mode: str, protection: str, known_received: int, all_ran: bool) -> Verdict:
    """
    Description: Decide the exit status of an agent run.
    Parameters:
        mode: direct or agent.
        protection: the path the target claims to protect: none, read-tool or gate.
        known_received: how many known-format canaries reached the mock.
        all_ran: whether every requested agent ran its whole script.
    Output: Verdict(1, ...) when a known-format canary was received and a protection is claimed; else Verdict(2, ...)
        when an agent was skipped or did not finish (never a silent pass); else Verdict(0, "ok").
    """
    if protection != "none" and known_received:
        return Verdict(1, f"{known_received} known-format canaries reached the provider through the {protection} path")
    if not all_ran:
        return Verdict(2, "inconclusive: an agent was skipped or did not finish its script")
    return Verdict(0, "ok")


def direct_verdict(planted: int, received: int) -> Verdict:
    """
    Description: Decide the exit status of a direct-mode run against the bare mock, which checks that the counter works.
    Parameters:
        planted: how many canaries the corpus planted.
        received: how many of them the recording holds.
    Output: Verdict(0, "ok") when every planted canary was received; otherwise Verdict(1, ...) naming the shortfall.
    """
    if received == planted:
        return Verdict(0, "ok")
    return Verdict(1, f"counter check failed: only {received} of {planted} planted canaries were found in the recording")


def assert_no_values(text: str, canaries: Iterable[Canary]) -> None:
    """
    Description: Check that no canary value, and no encoded form or long line of one, appears in a piece of text.
    Parameters:
        text: the serialized report.
        canaries: every planted canary.
    Output: None. Raises ReportLeakError naming the canary ident and form when one is found.
    """
    for canary in canaries:
        for form, needle in match.needles(canary.value):
            if needle in text:
                raise ReportLeakError(f"the report would contain the {form} form of canary {canary.ident}")


def write_report(path: Path, data: dict, canaries: Iterable[Canary]) -> Path:
    """
    Description: Write a report as JSON, after checking that it holds no canary value.
    Parameters:
        path: the file to write; its folder is created.
        data: the report.
        canaries: every planted canary, for the check.
    Output: The path. Raises ReportLeakError, writing nothing, when the serialized report contains a value.
    """
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert_no_values(text, list(canaries))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return Path(path)
