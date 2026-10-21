"""The synthetic request corpus for the egress gate (FR-PROOF-04, FR-GATE-02).

    from bench.gate_corpus import build_corpus
    cases = build_corpus(seed=0)

One request per block, item and tool type covered by FR-GATE-02 (docs/design.md), in both provider formats, each carrying one
generated canary in the field the spec says the gate scans. The canary harness sends the cases to the mock provider
(direct mode); the future test_fr_gate_02_* parser tests (G4) use the same cases, so a case's test_name is the spec's
name for it. A case has source "spec" when its shape is written from the spec tables and official docs, and "captured"
when a recorded agent session sent that block, item or tool type (CAPTURED, from the G2 probe run CAPTURED_RUN; see
bench/results/proof/g2-probes-<run>.json). A captured case still carries a generated canary, not what the agent held.

Every canary comes from bench/canary_repo.py's generators with a seed derived from (seed, case id). The same seed gives
the same corpus. A Case holds the canary value, so print a Case with summary(), never repr-ing its body into a report.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

from bench import canary_match as match
from bench import canary_repo

TEST_PREFIX = "test_fr_gate_02_"
ROUTES = {"anthropic": "/v1/messages", "responses": "/v1/responses"}
# Cases whose block, item or tool type Claude Code or Codex actually sent in the G2 probes (claude-corpus, codex-corpus).
CAPTURED_RUN = "20261019T175710Z-g2"
CAPTURED = frozenset({
    "anthropic_text", "anthropic_thinking", "anthropic_redacted_thinking", "anthropic_tool_use", "anthropic_tool_result_string",
    "anthropic_tool_result_blocks", "anthropic_system_role_text", "anthropic_system_role_string", "anthropic_tools_custom",
    "anthropic_system_blocks", "anthropic_metadata_user_id",
    "responses_input_text", "responses_output_text", "responses_function_call", "responses_function_call_output",
    "responses_reasoning", "responses_tools_function", "responses_tools_namespace", "responses_tools_builtin",
    "responses_client_metadata",
})
# Single-line kinds, alternating known-format and ML-only so both classes appear across the corpus.
CYCLE = ("github_token", "db_password", "openai_key", "aws_secret_access_key", "stripe_key", "twilio_auth_token",
         "slack_token", "postgres_password", "sendgrid_key", "db_url", "gcp_api_key", "aws_access_key_id", "npm_token",
         "jwt")


class Row(NamedTuple):
    """
    Description: The definition of one corpus case, before its canary is generated.
    Parameters:
        case_id: unique id; for a case the parser spec lists, the spec's test name without the test_fr_gate_02_ prefix.
        section: content-block, system-role, tools, top-level, input-item or policy.
        block_type: the block, item, tool or field this case exercises.
        expect: what a gate must do: redact (replace the canary), block (refuse the request) or pass (forward as is).
        build: a function from the canary value to the request body.
        kind: a canary kind to use instead of the rotating one (private_key for a multi-line value), or None.
        plant: False for a case with no scannable text (a pass-through type): the body is built without a canary.
        spec: True when the parser spec names a test for this case.
    Output: An immutable definition used by build_corpus.
    """

    case_id: str
    section: str
    block_type: str
    expect: str
    build: Callable[[str], dict]
    kind: Optional[str] = None
    plant: bool = True
    spec: bool = True


@dataclass(frozen=True)
class Case:
    """
    Description: One synthetic request and what a gate should do with it.
    Parameters:
        case_id: unique id (lower-case letters, digits, underscores).
        test_name: the parser spec's test name for the case, or None for an extra case.
        fmt: anthropic or responses.
        route: the path to send it to: /v1/messages or /v1/responses.
        section: content-block, system-role, tools, top-level, input-item or policy.
        block_type: the block, item, tool or field exercised.
        expect: redact, block or pass.
        canary_kind: the kind planted (a key of canary_repo's canaries), or None.
        canary_class: known-format or ml-only, or None.
        canary_value: the planted value; never put it in a report.
        field_path: where the canary sits in the body, as canary_match.locate names it, or None.
        body: the request body as a plain JSON-serializable dict.
        source: spec (written from the spec tables) or captured (the type was sent by an agent in a recorded session).
    Output: An immutable record; its repr and summary() hold no canary value.
    """

    case_id: str
    test_name: Optional[str]
    fmt: str
    route: str
    section: str
    block_type: str
    expect: str
    canary_kind: Optional[str]
    canary_class: Optional[str]
    canary_value: Optional[str] = field(repr=False)
    field_path: Optional[str]
    body: dict = field(repr=False)
    source: str = "spec"

    def summary(self) -> dict:
        """
        Description: Describe the case without its canary value or its body.
        Parameters: None
        Output: A dict with case_id, test_name, fmt, route, section, block_type, expect, canary_kind, canary_class,
            field_path and source.
        """
        return {"case_id": self.case_id, "test_name": self.test_name, "fmt": self.fmt, "route": self.route,
                "section": self.section, "block_type": self.block_type, "expect": self.expect,
                "canary_kind": self.canary_kind, "canary_class": self.canary_class, "field_path": self.field_path,
                "source": self.source}


def _canary(seed: int, case_id: str, kind: str) -> str:
    """
    Description: Generate the canary value for one case.
    Parameters:
        seed: the corpus seed.
        case_id: the case, which makes the value differ from the other cases' values.
        kind: the canary kind to return.
    Output: The generated value. Seeds the global random generator from (seed, case_id), as canary_repo does.
    """
    random.seed(f"{seed}:{case_id}")
    return canary_repo._canaries()[kind]


def build_corpus(seed: int = 0) -> list[Case]:
    """
    Description: Build every corpus case for a seed.
    Parameters:
        seed: the seed; the same seed gives the same cases and canary values.
    Output: The cases in a fixed order: Anthropic Messages first, then OpenAI Responses. A case with plant True has one
        canary in its body, found by canary_match.locate; its field_path is the first place it sits. Raises ValueError if
        a body does not hold its canary (a corpus bug).
    """
    from bench import (  # noqa: PLC0415 - they import Row from this module
        gate_corpus_anthropic,
        gate_corpus_responses,
    )

    cases: list[Case] = []
    rows = [("anthropic", r) for r in gate_corpus_anthropic.rows()] + [("responses", r) for r in gate_corpus_responses.rows()]
    planted = 0
    for fmt, row in rows:
        kind = value = path = None
        if row.plant:
            kind = row.kind or CYCLE[planted % len(CYCLE)]
            planted += 1
            value = _canary(seed, row.case_id, kind)
        body = row.build(value or "")
        if value is not None:
            hits = match.locate(json.dumps(body), value)
            if not hits:
                raise ValueError(f"corpus case {row.case_id} does not hold its canary")
            path = hits[0].path
        cases.append(Case(row.case_id, TEST_PREFIX + row.case_id if row.spec else None, fmt, ROUTES[fmt], row.section,
                          row.block_type, row.expect, kind, match.kind_class(kind) if kind else None, value, path, body,
                          "captured" if row.case_id in CAPTURED else "spec"))
    return cases
