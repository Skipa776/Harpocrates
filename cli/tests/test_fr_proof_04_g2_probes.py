"""FR-PROOF-04 (G2): the building blocks of the capture probes (bench/g2_probes.py), without running an agent.

The probes themselves run the real CLIs against the mock; they are run by hand for the G2 captures and write
bench/results/proof/g2-probes-<run>.json. These tests check the fakes they use and that nothing they write can hold a
credential-shaped value.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest
import tomllib

from bench import agent_isolation as iso
from bench import agent_runner as runner
from bench import g2_probes as probes
from bench import mock_provider as mock

URL = "http://127.0.0.1:5555"
RECOVERY_PHRASES = ("thinking", "signature", "bound to a different conversation", "Extra inputs are not permitted", "prompt is too long")


@pytest.fixture
def tw():
    throwaway = iso.make_throwaway()
    yield throwaway
    iso.discard(throwaway)


def test_fr_proof_04_g2_probes_run_process_can_capture_output(tmp_path: Path) -> None:
    code, timed_out, _, out, err = runner.run_process_captured(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper()); print('e', file=sys.stderr)"],
        {"PATH": os.environ["PATH"]}, tmp_path, 30, b"abc")
    assert (code, timed_out, out.strip(), err.strip()) == (0, False, "ABC", "e")
    assert runner.run_process_captured([sys.executable, "-c", "import time; time.sleep(30)"], {"PATH": os.environ["PATH"]},
                                       tmp_path, 1)[:2] == (None, True)


def test_fr_proof_04_g2_probes_fake_chatgpt_login_is_unmistakably_fake_and_never_expires_soon() -> None:
    auth = probes.fake_chatgpt_auth()
    assert auth["auth_mode"] == "chatgpt" and auth["OPENAI_API_KEY"] is None
    tokens = auth["tokens"]
    for name in ("id_token", "access_token"):
        header, payload, signature = tokens[name].split(".")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        assert claims["exp"] > 4_000_000_000  # no refresh is ever attempted
        assert signature == "fakesignature" and json.loads(base64.urlsafe_b64decode(header + "==")) == {"alg": "none", "typ": "JWT"}
    claims = json.loads(base64.urlsafe_b64decode(tokens["id_token"].split(".")[1] + "=="))
    assert claims["email"].endswith("@example.test") and "fake" in tokens["refresh_token"] and "fake" in tokens["account_id"]
    assert "fake" in json.dumps(auth)


def test_fr_proof_04_g2_probes_codex_configs_point_only_at_the_mock(tw) -> None:
    probes.write_route_b_config(tw, URL, chatgpt=False)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    assert config["openai_base_url"] == URL + "/v1" and "model_provider" not in config and "chatgpt_base_url" not in config
    assert config["features"]["plugins"] is False and config["analytics"]["enabled"] is False
    probes.write_route_b_config(tw, URL, chatgpt=True)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    assert config["chatgpt_base_url"] == URL + "/backend-api/" and config["openai_base_url"] == URL + "/v1"
    assert (tw.home / ".codex" / "auth.json").is_file() and (tw.home / ".codex" / "auth.json").stat().st_mode & 0o077 == 0
    runner.write_codex_config(tw, URL)
    probes.write_chatgpt_auth(tw, URL)
    config = tomllib.loads((tw.home / ".codex" / "config.toml").read_text())
    assert config["chatgpt_base_url"] == URL + "/backend-api/" and config["model_providers"]["mockprov"]["base_url"] == URL + "/v1"
    env = probes.chatgpt_env(tw, URL)
    assert env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"] == URL + "/oauth/token" and "CODEX_API_KEY" not in env
    assert env["CODEX_HOME"] == str(tw.home / ".codex")


def test_fr_proof_04_g2_probes_scripts_are_valid_and_carry_thinking(tw, tmp_path: Path) -> None:
    for name, script in (("claude", probes.claude_corpus_script(tw.repo)), ("codex", probes.codex_corpus_script())):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(script))
        assert mock.load_script(path) == script
        assert any(t.get("thinking") for t in script) and any(t.get("redacted_thinking") for t in script)
        assert script[-1] == {"text": "done"}
    names = [c["name"] for t in probes.claude_corpus_script(tw.repo) for c in t.get("tool_calls", [])]
    assert names == ["Read", "Bash", "Read", "mcp__harpo__safe_read"]
    assert (probes.claude_corpus_script(tw.repo)[2]["tool_calls"][0]["input"]["file_path"]).endswith("logo.png")


def test_fr_proof_04_g2_probes_the_block_message_avoids_every_recovery_phrase() -> None:
    message = probes.BLOCK_MESSAGE.lower()
    assert not any(phrase.lower() in message for phrase in RECOVERY_PHRASES)
    assert "harpo" in message and len(probes.BLOCK_MESSAGE) < 300


def test_fr_proof_04_g2_probes_long_prompts_are_big_and_plain() -> None:
    text = probes.long_prompt(300_000)
    assert 300_000 <= len(text) < 301_000 and text.isascii() and "\0" not in text


def test_fr_proof_04_g2_probes_png_fixture_is_a_real_png(tw) -> None:
    probes.write_png(tw.repo / "logo.png")
    data = (tw.repo / "logo.png").read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and data.endswith(b"IEND\xaeB`\x82")


def test_fr_proof_04_g2_probes_a_report_with_a_fake_credential_is_refused(tmp_path: Path) -> None:
    fakes = [iso.FAKE_API_KEY, iso.FAKE_OPENAI_KEY, probes.fake_chatgpt_auth()["tokens"]["refresh_token"]]
    probes.write_probe_report(tmp_path / "ok.json", {"probe": "x", "n": 1}, fakes)
    assert json.loads((tmp_path / "ok.json").read_text())["probe"] == "x"
    for fake in fakes:
        with pytest.raises(probes.ProbeReportError):
            probes.write_probe_report(tmp_path / "bad.json", {"oops": f"has {fake}"}, fakes)
    assert not (tmp_path / "bad.json").exists()
