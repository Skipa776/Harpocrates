"""LLM slot generation: deterministic jobs and strict output validation."""

from scripts.generate_llm_slots import clean_output, jobs


def test_jobs_are_deterministic_and_typed():
    a, b = jobs(50, seed=3), jobs(50, seed=3)
    assert a == b and len(a) == 50
    assert all(j["secret_kinds"] and j["nonsecret_kinds"] for j in a)
    assert len({j["language"] for j in a}) > 10


def test_clean_output_strips_fences_and_requires_markers():
    raw = '```python\nkey = "{{SECRET:stripe_key}}"\nsha = "{{NONSECRET:git_sha}}"\n```\n'
    assert clean_output(raw) == 'key = "{{SECRET:stripe_key}}"\nsha = "{{NONSECRET:git_sha}}"\n'
    assert clean_output("print('no slots here')") is None
    assert clean_output('x = "{{SECRET}}"') is None            # malformed
    assert clean_output('x = "{{SECRET:made_up}}"') is None    # unknown kind


def test_cli_commands_are_text_only(tmp_path):
    import json
    from types import SimpleNamespace

    from scripts.generate_llm_slots import _OPENCODE_TEXT_ONLY, _cli_command

    out = tmp_path / "last_message.txt"
    args = lambda cli: SimpleNamespace(cli=cli, model="m", effort="low", timeout=60)  # noqa: E731
    codex = _cli_command(args("codex"), "p", out)
    assert codex[codex.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in codex  # no user MCP servers, plugins, or hooks
    claude = _cli_command(args("claude"), "p", out)
    assert claude[claude.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in claude  # --tools "" alone leaves user MCP tools enabled
    assert json.loads(claude[claude.index("--mcp-config") + 1]) == {"mcpServers": {}}
    opencode = _cli_command(args("opencode"), "p", out)
    assert opencode[opencode.index("--dir") + 1] == str(tmp_path)
    assert "--pure" in opencode
    agy = _cli_command(args("agy"), "p", out)
    assert agy[agy.index("--mode") + 1] == "plan" and "--sandbox" in agy
    assert "--dangerously-skip-permissions" not in agy
    config = json.loads(_OPENCODE_TEXT_ONLY)
    assert not any(config["tools"].values())
    assert all(v == "deny" for v in config["permission"].values())
    assert not any(server["enabled"] for server in config["mcp"].values())
