"""FR-INSTALL-01, FR-INSTALL-08, FR-UX-01 and SEC-03: `harpo guard up` and `harpo guard down`, called as functions.

Every test runs in a temp home (the `home` fixture). No test starts `harpo` as a subprocess.
"""
from __future__ import annotations

import json
import os
import stat

import pytest
from toml_compat import tomllib

from Harpocrates import guard, install
from Harpocrates.gate import session_token


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    (h / ".codex").mkdir()
    monkeypatch.setenv("HOME", str(h))
    yield h
    guard.stop_gate(guard.Files(h / ".harpo"))  # never leave a gate running


def _settings(h):
    return h / ".claude" / "settings.json"


def _config(h):
    return h / ".codex" / "config.toml"


def _set(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def _json(path):
    return json.loads(path.read_text())


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _snapshot(h):
    """Bytes of both agent files, or None when a file does not exist."""
    return {p: (p.read_bytes() if p.exists() else None) for p in (_settings(h), _config(h))}


BASE_ENV = {"ANTHROPIC_BASE_URL": "https://api.example.com"}


def test_fr_install_01_guard_up_routes_claude_code_and_codex_to_a_running_gate(home):
    files = guard.Files(home / ".harpo")
    out = guard.up(["claude-code", "codex"], port=0)
    assert not out.failed, out.lines
    pid, port = guard.running_gate(files)
    assert _alive(pid)
    token = session_token.read_token_file(files.token)

    env = _json(_settings(home))["env"]
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{port}"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == f"{guard.TOKEN_HEADER}: {token}"

    cfg = tomllib.loads(_config(home).read_text())
    assert cfg["model_provider"] == "harpo"
    prov = cfg["model_providers"]["harpo"]
    assert prov["base_url"] == f"http://127.0.0.1:{port}/v1"
    assert prov["wire_api"] == "responses"
    assert prov["requires_openai_auth"] is True
    assert prov["http_headers"] == {guard.TOKEN_HEADER: token}

    record = _json(files.record)["gate"]
    assert record["enabled"] is True
    assert record["port"] == port
    assert record["token_header"] == guard.TOKEN_HEADER
    assert _json(files.pid) == {"pid": pid, "port": port}


@pytest.mark.parametrize(
    "name, value",
    [
        ("ANTHROPIC_BASE_URL", "https://proxy.example/v1"),
        ("ANTHROPIC_CUSTOM_HEADERS", "x-mine: 1"),
    ],
)
def test_fr_install_01_guard_up_refuses_a_base_url_or_custom_headers_harpo_did_not_write(home, name, value):
    files = guard.Files(home / ".harpo")
    _set(_settings(home), json.dumps({"env": {name: value}}, indent=2) + "\n")
    before = _snapshot(home)

    out = guard.up(["claude-code", "codex"], port=0)

    assert out.failed
    assert _snapshot(home)[_settings(home)] == before[_settings(home)]
    # Codex is still routed to the gate.
    assert "harpo" in tomllib.loads(_config(home).read_text())["model_providers"]
    assert guard.running_gate(files) is not None


@pytest.mark.parametrize("value, refused", [("1", True), ("0", False)])
def test_fr_install_01_guard_up_refuses_a_provider_switch_that_is_on(home, value, refused):
    files = guard.Files(home / ".harpo")
    _set(_settings(home), json.dumps({"env": {"CLAUDE_CODE_USE_BEDROCK": value}}, indent=2) + "\n")
    before = _settings(home).read_bytes()

    out = guard.up(["claude-code"], port=0)

    if refused:
        assert out.failed
        assert _settings(home).read_bytes() == before
        assert guard.running_gate(files) is None
    else:
        assert not out.failed, out.lines
        assert _json(_settings(home))["env"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    guard.stop_gate(files)


@pytest.mark.parametrize(
    "text",
    [
        'model_provider = "x"\n',
        'openai_base_url = "https://p.example/v1"\n',
        '[model_providers.harpo]\nname = "mine"\n',
    ],
    ids=["model_provider", "openai_base_url", "harpo_table"],
)
def test_fr_install_01_guard_up_refuses_a_codex_provider_already_set(home, text):
    files = guard.Files(home / ".harpo")
    _set(_config(home), text)
    before = _config(home).read_bytes()

    out = guard.up(["claude-code", "codex"], port=0)

    assert out.failed
    assert _config(home).read_bytes() == before
    # Claude Code is still routed to the gate.
    assert _json(_settings(home))["env"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    guard.stop_gate(files)


def test_fr_install_01_with_every_harness_refused_no_gate_starts(home):
    files = guard.Files(home / ".harpo")
    _set(_settings(home), json.dumps({"env": BASE_ENV}, indent=2) + "\n")
    _set(_config(home), 'model_provider = "x"\n')
    before = _snapshot(home)

    out = guard.up(["claude-code", "codex"], port=0)

    assert out.failed
    assert _snapshot(home) == before
    assert not files.pid.exists()
    assert guard.running_gate(files) is None
    if files.record.exists():
        assert "gate" not in _json(files.record)


def test_fr_install_01_dry_run_starts_and_writes_nothing(home):
    files = guard.Files(home / ".harpo")
    _set(_settings(home), json.dumps({"theme": "dark"}, indent=2) + "\n")
    before = _snapshot(home)

    out = guard.up(["claude-code", "codex"], port=0, dry_run=True)

    assert not out.failed, out.lines
    assert not files.pid.exists()
    assert guard.running_gate(files) is None
    assert _snapshot(home) == before
    assert not files.record.exists()


def test_fr_install_08_guard_down_restores_both_files_byte_identical(home):
    settings_text = '{\n    "theme": "dark"\n}'  # 4-space indent, no final newline
    config_text = 'model = "gpt-5"\n\n[profile.work]\napproval_policy = "never"\n'
    _set(_settings(home), settings_text)
    _set(_config(home), config_text)

    assert not guard.up(["claude-code", "codex"], port=0).failed
    assert _settings(home).read_bytes() != settings_text.encode()

    out = guard.down()

    assert not out.failed, out.lines
    assert _settings(home).read_bytes() == settings_text.encode()
    assert _config(home).read_bytes() == config_text.encode()


def test_fr_install_08_guard_down_deletes_files_guard_up_created(home):
    assert not _settings(home).exists()
    assert not _config(home).exists()

    assert not guard.up(["claude-code", "codex"], port=0).failed
    assert _settings(home).exists()
    assert _config(home).exists()

    out = guard.down()

    assert not out.failed, out.lines
    assert not _settings(home).exists()
    assert not _config(home).exists()


def test_fr_install_08_an_edit_after_guard_up_is_kept(home):
    files = guard.Files(home / ".harpo")
    assert not guard.up(["claude-code"], port=0).failed

    data = _json(_settings(home))
    data["theme"] = "dark"
    _set(_settings(home), json.dumps(data, indent=2) + "\n")

    out = guard.down()

    assert not out.failed, out.lines
    after = _json(_settings(home))
    assert after["theme"] == "dark"
    assert not any(k.startswith("ANTHROPIC_") for k in after.get("env", {}))
    assert guard.running_gate(files) is None


def test_fr_install_08_up_twice_then_down_restores_the_original(home):
    settings_text = '{\n    "theme": "dark"\n}'
    config_text = 'model = "gpt-5"\n'
    _set(_settings(home), settings_text)
    _set(_config(home), config_text)

    assert not guard.up(["claude-code", "codex"], port=0).failed
    assert not guard.up(["claude-code", "codex"], port=0).failed
    assert not guard.down().failed

    assert _settings(home).read_bytes() == settings_text.encode()
    assert _config(home).read_bytes() == config_text.encode()


def test_fr_ux_01_up_twice_rotates_the_token_and_replaces_the_gate(home):
    files = guard.Files(home / ".harpo")
    assert not guard.up(["claude-code", "codex"], port=0).failed
    token1 = session_token.read_token_file(files.token)
    old_pid, _ = guard.running_gate(files)

    assert not guard.up(["claude-code", "codex"], port=0).failed
    token2 = session_token.read_token_file(files.token)
    new_pid, _ = guard.running_gate(files)

    assert token2 != token1
    assert new_pid != old_pid
    assert not _alive(old_pid) or guard.running_gate(files)[0] != old_pid


def test_fr_install_08_no_backup_holds_the_token(home):
    files = guard.Files(home / ".harpo")
    _set(_settings(home), json.dumps({"env": {"KEEP": "1"}}, indent=2) + "\n")
    _set(_config(home), 'model = "gpt-5"\n')

    assert not guard.up(["claude-code", "codex"], port=0).failed
    token = session_token.read_token_file(files.token)
    assert not guard.down().failed

    # Any *.harpo.bak that up made holds the original file, never the token.
    needle = token.encode()
    leaks = [p for p in home.rglob("*") if p.is_file() and needle in p.read_bytes()]
    assert leaks == []


def test_sec_03_gate_files_are_private(home):
    files = guard.Files(home / ".harpo")
    assert not guard.up(["claude-code", "codex"], port=0).failed

    for path in (files.token, files.pid, files.log):
        assert path.exists(), path
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, path
    guard.stop_gate(files)


def test_fr_install_01_harpo_install_keeps_the_gate_entry(home):
    files = guard.Files(home / ".harpo")
    files.state.mkdir(exist_ok=True)
    gate = {"enabled": True, "port": 47321, "token_header": guard.TOKEN_HEADER, "routes": {}}
    files.record.write_text(json.dumps({"version": 1, "gate": gate}, indent=2) + "\n")

    install.install_harnesses(["codex"])
    assert _json(files.record)["gate"] == gate

    install.uninstall_harnesses(["codex"])
    assert _json(files.record)["gate"] == gate
