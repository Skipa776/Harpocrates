"""FR-PROOF-04: the mock provider (bench/mock_provider.py) records what an agent sends, never a credential.

Everything here runs against 127.0.0.1 only. The credential values are made-up strings.
"""
from __future__ import annotations

import pytest

from bench import mock_provider as mock

FAKE = "FAKE-CRED-0123456789abcdef"


@pytest.mark.parametrize("name", [
    "Authorization", "authorization", "Proxy-Authorization", "x-api-key", "X-Api-Key", "api-key", "openai-api-key",
    "Cookie", "Set-Cookie", "x-auth-token", "X-Amz-Security-Token", "x-claude-code-session-id", "x-session-affinity",
    "x-client-secret", "ChatGPT-Account-ID", "chatgpt-account-id", "x-goog-api-key", "anthropic-auth-bearer",
])
def test_fr_proof_04_mock_redacts_every_credential_header_name(name: str) -> None:
    assert mock.is_credential_header(name)
    assert mock.redact_headers([(name, FAKE)]) == [[name, "<present>"]]


@pytest.mark.parametrize("name", [
    "anthropic-version", "anthropic-beta", "content-type", "content-length", "content-encoding", "user-agent",
    "accept", "host", "x-stainless-lang", "x-codex-turn-state", "x-codex-turn-metadata", "openai-beta", "version",
    "x-client-request-id", "transfer-encoding", "connection", "accept-encoding",
])
def test_fr_proof_04_mock_keeps_other_headers_verbatim(name: str) -> None:
    assert not mock.is_credential_header(name)
    assert mock.redact_headers([(name, "some-value")]) == [[name, "some-value"]]


def test_fr_proof_04_mock_redaction_keeps_order_and_duplicates() -> None:
    got = mock.redact_headers([("A", "1"), ("Cookie", FAKE), ("A", "2"), ("x-api-key", FAKE)])
    assert got == [["A", "1"], ["Cookie", "<present>"], ["A", "2"], ["x-api-key", "<present>"]]
    assert FAKE not in repr(got)


@pytest.mark.parametrize("target, expected", [
    ("/v1/messages?beta=true", "/v1/messages?beta=true"),
    ("/v1/models?limit=1000", "/v1/models?limit=1000"),
    ("/v1/messages?key=" + FAKE, "/v1/messages?key=%3Cpresent%3E"),
    ("/x?a=1&api_key=" + FAKE + "&token=" + FAKE + "&b=2", "/x?a=1&api_key=%3Cpresent%3E&token=%3Cpresent%3E&b=2"),
    ("/x?session_id=" + FAKE, "/x?session_id=%3Cpresent%3E"),
])
def test_fr_proof_04_mock_redacts_credential_query_values(target: str, expected: str) -> None:
    got = mock.redact_target(target)
    assert got == expected and FAKE not in got


# --- decoding the request body for the recording ---------------------------------------------------------------

import gzip  # noqa: E402
import zlib  # noqa: E402


def test_fr_proof_04_mock_decodes_identity_and_absent_encoding() -> None:
    for encoding in (None, "", "identity", "IDENTITY"):
        assert mock.decode_body(b'{"a":1}', encoding, 1024) == mock.Decoded(b'{"a":1}', "ok")


def test_fr_proof_04_mock_decodes_gzip() -> None:
    raw = gzip.compress(b'{"hello": "world"}')
    assert mock.decode_body(raw, "gzip", 1024) == mock.Decoded(b'{"hello": "world"}', "ok")
    assert mock.decode_body(raw, "GZIP", 1024).status == "ok"


def test_fr_proof_04_mock_gzip_bomb_is_too_large_not_expanded() -> None:
    bomb = gzip.compress(b"\0" * (8 * 1024 * 1024))
    got = mock.decode_body(bomb, "gzip", 1024 * 1024)
    assert got.status == "too-large" and got.body is None


def test_fr_proof_04_mock_identity_over_the_cap_is_too_large() -> None:
    assert mock.decode_body(b"x" * 11, None, 10) == mock.Decoded(None, "too-large")
    assert mock.decode_body(b"x" * 10, None, 10).status == "ok"


@pytest.mark.parametrize("raw", [b"this is not gzip", gzip.compress(b"abcdef" * 50)[:-6], b"", b"\x1f\x8b\x08\x00junk"])
def test_fr_proof_04_mock_wrong_or_truncated_gzip_is_bad_encoding(raw: bytes) -> None:
    got = mock.decode_body(raw, "gzip", 1024)
    assert got == mock.Decoded(None, "bad-encoding")


@pytest.mark.parametrize("encoding", ["br", "deflate", "gzip, zstd", "compress", "unknown"])
def test_fr_proof_04_mock_other_encodings_are_unsupported(encoding: str) -> None:
    assert mock.decode_body(zlib.compress(b"x"), encoding, 1024) == mock.Decoded(None, "unsupported-encoding")


def test_fr_proof_04_mock_zstd_is_flagged_when_no_module_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: None)
    assert mock.decode_body(b"\x28\xb5\x2f\xfd\x00", "zstd", 1024) == mock.Decoded(None, "zstd-unavailable")


def test_fr_proof_04_mock_zstd_is_decoded_when_a_module_is_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock, "_load_zstd", lambda: lambda data, cap: b'{"z": 1}')
    assert mock.decode_body(b"fake-zstd-frame", "zstd", 1024) == mock.Decoded(b'{"z": 1}', "ok")

    def fail(data: bytes, cap: int) -> bytes:
        raise ValueError("corrupt frame")

    monkeypatch.setattr(mock, "_load_zstd", lambda: fail)
    assert mock.decode_body(b"x", "zstd", 1024) == mock.Decoded(None, "bad-encoding")
    monkeypatch.setattr(mock, "_load_zstd", lambda: lambda data, cap: b"x" * (cap + 1))
    assert mock.decode_body(b"x", "zstd", 8) == mock.Decoded(None, "too-large")


# --- the scripted replies ------------------------------------------------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402


def write_script(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(payload))
    return path


def test_fr_proof_04_mock_script_loads_a_list_of_turns(tmp_path: Path) -> None:
    turns = [{"tool_calls": [{"name": "Read", "input": {"file_path": ".env"}}]},
             {"text": "done"}, {"error": {"status": 400, "message": "blocked"}}]
    assert mock.load_script(write_script(tmp_path, turns)) == turns


@pytest.mark.parametrize("payload", [
    {"turns": []}, "text", [1], [{"surprise": 1}], [{"text": 5}], [{"tool_calls": "Read"}], [{"tool_calls": [{}]}],
    [{"tool_calls": [{"name": "Read", "input": []}]}], [{"error": {"status": 200, "message": "x"}}],
    [{"error": {"status": "400", "message": "x"}}], [{"error": {"status": 400}}],
])
def test_fr_proof_04_mock_script_rejects_a_bad_shape(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ValueError):
        mock.load_script(write_script(tmp_path, payload))


def test_fr_proof_04_mock_script_rejects_unreadable_or_non_json_files(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError):
        mock.load_script(tmp_path / "bad.json")
    with pytest.raises(ValueError):
        mock.load_script(tmp_path / "missing.json")


def test_fr_proof_04_mock_counts_tool_results_in_an_anthropic_body() -> None:
    body = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a"}, {"type": "text", "text": "y"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "b"}]}]},
    ]}
    assert mock.count_tool_results(body, "anthropic") == 2
    assert mock.count_tool_results({"messages": [{"role": "user", "content": "hi"}]}, "anthropic") == 0
    assert mock.count_tool_results({}, "anthropic") == 0 and mock.count_tool_results([1], "anthropic") == 0


def test_fr_proof_04_mock_counts_tool_outputs_in_a_responses_body() -> None:
    body = {"input": [
        {"type": "message", "role": "user", "content": "go"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "a"},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": "b"},
        {"type": "local_shell_call_output", "id": "c3", "output": "c"},
    ]}
    assert mock.count_tool_results(body, "responses") == 3
    assert mock.count_tool_results({"input": "just text"}, "responses") == 0
    assert mock.count_tool_results({}, "responses") == 0


def test_fr_proof_04_mock_picks_the_turn_by_tool_results_already_in_the_request() -> None:
    turns = [{"tool_calls": [{"name": "Read"}]}, {"tool_calls": [{"name": "Bash"}, {"name": "Grep"}]}, {"text": "done"}]
    assert mock.select_turn(turns, 0) is turns[0]
    assert mock.select_turn(turns, 1) is turns[1]  # the first call has an answer
    assert mock.select_turn(turns, 3) is turns[2]  # both calls of turn 2 have answers
    assert mock.select_turn(turns, 2) is turns[2]  # between turns: the next one that starts at or after 2
    assert mock.select_turn(turns, 99) is turns[2]  # past the end: the last turn, again and again
    assert mock.select_turn([], 5) == {}


def test_fr_proof_04_mock_a_request_without_tools_gets_a_plain_text_turn() -> None:
    turns = [{"tool_calls": [{"name": "Read"}]}]
    assert mock.turn_for({"messages": [], "tools": [{"name": "Read"}]}, "anthropic", turns) is turns[0]
    assert mock.turn_for({"messages": []}, "anthropic", turns) == {}
    assert mock.turn_for({"messages": [], "tools": []}, "anthropic", turns) == {}
    assert mock.turn_for({"input": "x", "tools": [{"type": "function"}]}, "responses", turns) is turns[0]
    assert mock.turn_for({"input": "x"}, "responses", turns) == {}
    assert mock.turn_for("not an object", "responses", turns) == {}


# --- no header outside the redaction set can hold a credential -------------------------------------------------------

import glob  # noqa: E402

# Header names that the agents were seen to send (G2 probe run 20261019T175710Z-g2) and that are not redacted. Each one
# carries a version, an id, a content-negotiation value or a fixed flag; none is a credential. A header that is neither
# credential-named nor in this set makes the test below fail, so a new agent version cannot slip a new header, whose value
# might be a credential, into the recording unreviewed.
BENIGN_HEADERS = frozenset({
    "accept", "accept-encoding", "anthropic-beta", "anthropic-dangerous-direct-browser-access", "anthropic-version",
    "cache-control", "connection", "content-encoding", "content-length", "content-type", "host", "openai-beta", "originator", "thread-id",
    "transfer-encoding", "upgrade", "user-agent", "version", "x-app", "x-client-request-id", "x-codex-beta-features",
    "x-codex-routing-hint", "x-codex-turn-metadata", "x-codex-turn-state", "x-codex-window-id",
})
BENIGN_PREFIXES = ("x-stainless-", "sec-websocket-")


def test_fr_proof_04_mock_no_header_outside_the_redaction_set_could_hold_a_credential() -> None:
    reports = sorted(glob.glob(str(Path(mock.__file__).resolve().parent / "results" / "proof" / "g2-probes-*.json")))
    assert reports, "the committed G2 probe report is missing"
    seen: set[str] = set()
    for path in reports:
        for probe in json.loads(Path(path).read_text())["probes"]:
            transport = probe.get("observations", {}).get("transport")
            if transport:
                seen |= set(transport["header_names"])
    assert len(seen) > 20
    kept = {name for name in seen if not mock.is_credential_header(name)}
    unreviewed = {name for name in kept if name not in BENIGN_HEADERS and not name.startswith(BENIGN_PREFIXES)}
    assert not unreviewed, f"review these headers, then add them to BENIGN_HEADERS or to the redaction words: {sorted(unreviewed)}"
    # And the ones that were redacted really are credential-named: the session ids, keys and tokens among them.
    redacted = seen - kept
    assert {"x-api-key", "authorization", "session-id", "x-claude-code-session-id", "chatgpt-account-id"} <= redacted
