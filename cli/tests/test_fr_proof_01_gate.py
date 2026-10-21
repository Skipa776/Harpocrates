"""FR-PROOF-01 at the gate: one proof entry per forwarded request, findings for redactions and blocks, fail closed."""
from __future__ import annotations

import json
import os
from pathlib import Path

from gate_canaries import known_format
from gate_rig import ANTHROPIC, JSON, MESSAGES, running_rig

from Harpocrates import audit


def _findings(state: Path) -> list[dict]:
    """The findings log lines, or [] when nothing was written."""
    path = state / "findings.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _secret_body(value: str) -> bytes:
    """A /v1/messages body that carries one secret value."""
    return json.dumps({"model": "m", "max_tokens": 1,
                       "messages": [{"role": "user", "content": f"key={value}"}]}).encode()


def test_fr_proof_01_two_requests_chain_the_forwarded_bodies(tmp_path: Path) -> None:
    """Two forwarded requests give two chained entries whose hashes and sizes match what the mock received."""
    with running_rig(audit_dir=tmp_path) as rig:
        for _ in range(2):
            rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        records = rig.mock.records()
    result = audit.verify(tmp_path)
    assert result.problem is None
    assert [e["seq"] for e in result.entries] == [1, 2]
    assert len(records) == 2
    for entry, record in zip(result.entries, records):
        assert entry["sha256"] == record["body_sha256"]
        assert entry["bytes"] == record["body_size"]
        assert entry["provider"] == "anthropic"
        assert entry["route"] == "anthropic.messages"


def test_fr_proof_01_placeholder_counts_are_per_type(tmp_path: Path) -> None:
    """A secret that appears twice is forwarded as placeholders, and the entry counts two of that type."""
    kind, value = sorted(known_format().items())[0]
    body = json.dumps({"model": "m", "max_tokens": 1, "messages": [{"role": "user",
                       "content": f"key={value} and again {value}"}]}).encode()
    with running_rig(audit_dir=tmp_path) as rig:
        rig.call("POST", "/v1/messages", body, {**JSON, **ANTHROPIC})
        record = rig.mock.records()[0]
    kinds = [secret_type for secret_type, _ in audit.placeholders(record["body"])]
    assert len(kinds) == 2 and len(set(kinds)) == 1
    leaked = value in record["body"]
    assert not leaked
    assert audit.verify(tmp_path).entries[0]["placeholders"] == {kinds[0]: 2}


def test_fr_proof_01_refused_requests_write_no_entry(tmp_path: Path) -> None:
    """A request with a wrong token (401) or an unknown path (400) is not forwarded, so it gets no entry."""
    with running_rig(audit_dir=tmp_path) as rig:
        wrong = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC}, token="not-the-token")
        unknown = rig.call("POST", "/v1/nope", MESSAGES, {**JSON, **ANTHROPIC})
        assert rig.mock.records() == []
    assert wrong.status == 401
    assert unknown.status == 400
    result = audit.verify(tmp_path)
    assert result.problem is None
    assert result.entries == []


def test_fr_proof_01_same_secret_twice_is_one_finding(tmp_path: Path) -> None:
    """The same secret in two requests gives one secret_redacted line, and no canary value reaches either log."""
    kind, value = sorted(known_format().items())[0]
    with running_rig(audit_dir=tmp_path) as rig:
        rig.call("POST", "/v1/messages", _secret_body(value), {**JSON, **ANTHROPIC})
        rig.call("POST", "/v1/messages", _secret_body(value), {**JSON, **ANTHROPIC})
    redacted = [line for line in _findings(tmp_path) if line["event_type"] == "secret_redacted"]
    assert len(redacted) == 1
    assert redacted[0]["layer"] == "gate"
    assert redacted[0]["source"] == "anthropic.messages"
    assert len(redacted[0]["hash"]) == 4
    assert all(c in "0123456789abcdef" for c in redacted[0]["hash"])
    leaked = any(value in (tmp_path / name).read_text(encoding="utf-8") for name in ("findings.jsonl", "proof.jsonl"))
    assert not leaked


def test_fr_proof_01_existing_placeholder_is_forwarded_without_finding(tmp_path: Path) -> None:
    """A placeholder already in the request is forwarded and counted, but it is not this layer's finding."""
    body = json.dumps({"model": "m", "max_tokens": 1,
                       "messages": [{"role": "user", "content": "<<HARPO:api_token:abcd>>"}]}).encode()
    with running_rig(audit_dir=tmp_path) as rig:
        reply = rig.call("POST", "/v1/messages", body, {**JSON, **ANTHROPIC})
        assert len(rig.mock.records()) == 1
    assert reply.status not in (400, 401)
    assert audit.verify(tmp_path).entries[0]["placeholders"] == {"api_token": 1}
    assert [line for line in _findings(tmp_path) if line["event_type"] == "secret_redacted"] == []


def test_fr_proof_01_blocked_route_with_token_is_one_finding(tmp_path: Path) -> None:
    """A blocked route with the token logs one request_blocked line; without the token it logs nothing."""
    with running_rig(audit_dir=tmp_path) as rig:
        rig.call("POST", "/v1/nope", MESSAGES, {**JSON, **ANTHROPIC})
        blocked = _findings(tmp_path)
        rig.call("POST", "/v1/nope", MESSAGES, {**JSON, **ANTHROPIC}, token=False)
        after = _findings(tmp_path)
    assert len(blocked) == 1
    assert blocked[0]["event_type"] == "request_blocked"
    assert blocked[0]["reason"] == "HARPO-BLOCK-ROUTE"
    assert blocked[0]["action"] == "block"
    assert after == blocked


def test_fr_proof_01_no_audit_dir_forwards_and_writes_nothing(tmp_path: Path) -> None:
    """Without audit_dir the request is forwarded and no audit file is written anywhere."""
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with running_rig() as rig:
        reply = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        assert len(rig.mock.records()) == 1
    assert reply.status not in (400, 401)
    assert list(fresh.iterdir()) == []
    assert not (Path(os.environ["HOME"]) / ".harpo").exists()


def test_fr_proof_01_unusable_audit_folder_fails_closed(tmp_path: Path) -> None:
    """An audit folder that is a symlink blocks the request with HARPO-BLOCK-AUDIT before anything is sent."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "audit"
    link.symlink_to(target)
    with running_rig(audit_dir=link) as rig:
        reply = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        assert rig.mock.records() == []
    assert reply.status == 400
    assert b"HARPO-BLOCK-AUDIT" in reply.body


def test_fr_proof_01_damaged_proof_head_fails_closed(tmp_path: Path) -> None:
    """A damaged proof.head next to an existing proof line blocks the next request with HARPO-BLOCK-AUDIT."""
    with running_rig(audit_dir=tmp_path) as rig:
        first = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        assert first.status not in (400, 401)
        (tmp_path / "proof.head").write_text("{}", encoding="utf-8")
        reply = rig.call("POST", "/v1/messages", MESSAGES, {**JSON, **ANTHROPIC})
        assert len(rig.mock.records()) == 1
    assert reply.status == 400
    assert b"HARPO-BLOCK-AUDIT" in reply.body
