"""Per-harness support tiers (FR-INSTALL-03) and read-tool setup config (FR-READ-03).

The config only denies the harness's built-in reads of secret-bearing paths and registers the
harpocrates MCP server, whose safe_read and safe_grep return those files redacted. Neither harness
applies its read-deny rules to MCP servers, so safe_read can still open what the built-in tools can't.
`harpocrates setup <harness>` prints it; writing it into the harness's own files is FR-INSTALL-01 (M2).
"""
from __future__ import annotations

import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# File names that hold secrets far more often than not, matched at any depth.
SENSITIVE_FILES = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ecdsa*", "id_ed25519*",
    "*.tfvars", "*.tfstate", "*.jks", "*.keystore", "credentials.json", ".netrc", ".npmrc", ".pypirc",
    ".git-credentials",
)
# Credential stores in the home directory, blocked as a whole.
SENSITIVE_HOME_DIRS = ("~/.ssh", "~/.aws", "~/.azure", "~/.config/gcloud", "~/.kube", "~/.docker", "~/.gnupg")

MCP_TOOLS = ("mcp__harpocrates__safe_read", "mcp__harpocrates__safe_grep")
AGENT_NOTE = (
    "Files that usually hold secrets (.env, keys, credential stores) are blocked for the built-in read "
    "tools. Read them with the harpocrates MCP tools safe_read and safe_grep; secrets come back as "
    "<<HARPO:type:hash>> placeholders. Never try to recover a placeholder's value."
)
# ponytail: flip when the egress gate ships (M2); until then no harness has more than the read tool.
GATE_AVAILABLE = False


@dataclass(frozen=True)
class Harness:
    name: str
    tier: str
    commands: tuple[str, ...]  # on PATH
    paths: tuple[str, ...]  # under home, or under the applications dir for *.app
    gap: str  # what stays unprotected even at this harness's full tier, printed after "Not protected:"
    setup: Optional[str] = None  # `harpocrates setup <setup>` prints its config


HARNESSES = (
    Harness("Claude Code", "Full", ("claude",), (".claude",),
            "Traffic outside the API base URL (telemetry, sign-in) and transcripts saved on disk.",
            setup="claude-code"),
    Harness("Codex CLI", "Full", ("codex",), (".codex",),
            "Traffic outside the API base URL (sign-in, analytics) and session files saved on disk.",
            setup="codex"),
    Harness("OpenCode", "Full, per provider", ("opencode",), (".config/opencode", ".opencode"),
            "Providers not pointed at the gate, and any provider a repo's own opencode.json switches to."),
    Harness("GitHub Copilot (own API key)", "Partial", (), (".vscode/extensions/github.copilot-chat*",),
            "Inline completions and Copilot-hosted models, which never pass through the gate. Only chat "
            "and agent requests to your own key can be scanned."),
    Harness("Cursor", "Read tool only", ("cursor",), (".cursor", "Cursor.app"),
            "Agent, Composer, and tab completion requests go to Cursor's servers unscanned, "
            "and so does anything the agent reads through its terminal. Even with a setup, only reads "
            "through safe_read would be redacted."),
)


def detect(home: Optional[Path] = None, which: Callable[[str], Optional[str]] = shutil.which,
           apps: Path = Path("/Applications")) -> dict[str, bool]:
    """Which harnesses look installed: a command on PATH, or a config or app directory."""
    home = home or Path.home()

    def present(rel: str) -> bool:
        base = apps if rel.endswith(".app") else home
        return any(base.glob(rel)) if "*" in rel else (base / rel).exists()

    return {h.name: any(which(c) for c in h.commands) or any(present(p) for p in h.paths) for h in HARNESSES}


def tier_report(found: dict[str, bool]) -> str:
    """Support tier per detected harness, with what each tier leaves unprotected (FR-INSTALL-03)."""
    rows = [h for h in HARNESSES if found.get(h.name)]
    if not rows:
        return "No supported harness detected (Claude Code, Codex CLI, OpenCode, Copilot, Cursor)."
    lines = ["Detected harnesses and their Harpocrates support tier:", ""]
    for h in rows:
        lines.append(f"  {h.name}: {h.tier}")
        lines.append(f"    Not protected: {h.gap}")
        if h.setup:
            lines.append(f"    Setup: harpocrates setup {h.setup}")
        else:
            lines.append("    Protected today: nothing. There is no setup for this harness yet.")
    if not GATE_AVAILABLE:
        lines += ["", "The egress gate isn't released yet (milestone M2). Until it is, every harness, Full tier "
                  "included, is protected by the read tool only: a secret the agent reaches through the "
                  "shell (cat, env, test output, git log -p) still goes to the model."]
    return "\n".join(lines)


def mcp_server() -> tuple[str, list[str]]:
    """This interpreter running the MCP server, so it works from any venv and from IDE-launched agents."""
    return sys.executable, ["-m", "Harpocrates.mcp.server"]


def claude_code_settings() -> dict:
    """Lines for ~/.claude/settings.json. `//**/` anchors at the filesystem root, so a rule in user
    settings covers every project and any file outside the working directory."""
    deny = [f"Read(//**/{name})" for name in SENSITIVE_FILES] + [f"Read({d}/**)" for d in SENSITIVE_HOME_DIRS]
    command, _ = mcp_server()
    # Content-aware: also blocks built-in reads of any other file that holds a secret (Harpocrates/hooks.py).
    hook = {"type": "command", "command": shlex.join([command, "-m", "Harpocrates.hooks"])}
    return {"permissions": {"deny": deny, "allow": list(MCP_TOOLS)},
            "hooks": {"PreToolUse": [{"matcher": "Read|Grep|Bash", "hooks": [hook]}]}}


def codex_config() -> str:
    """Lines for ~/.codex/config.toml. The profile limits sandboxed shell commands; Codex doesn't apply
    profiles to MCP servers, so safe_read still reads these paths. The PreToolUse hook is the same
    content-aware check as Claude Code's: it refuses shell commands that would print a file holding a secret."""
    command, args = mcp_server()
    files = "\n".join(f'"**/{name}" = "deny"' for name in SENSITIVE_FILES)
    dirs = "\n".join(f'"{d}" = "deny"' for d in SENSITIVE_HOME_DIRS)
    return f"""default_permissions = "harpocrates"

[mcp_servers.harpocrates]
command = {json.dumps(command, ensure_ascii=False)}
args = {json.dumps(args, ensure_ascii=False)}

[permissions.harpocrates]
description = "Workspace access with secret-bearing files denied; read them through safe_read."
extends = ":workspace"

[permissions.harpocrates.filesystem]
glob_scan_max_depth = 8
{dirs}

[permissions.harpocrates.filesystem.":workspace_roots"]
{files}

[[hooks.PreToolUse]]
matcher = "^Bash$"

[[hooks.PreToolUse.hooks]]
type = "command"
command = {json.dumps(shlex.join([command, "-m", "Harpocrates.hooks"]), ensure_ascii=False)}
statusMessage = "Harpocrates: checking files for secrets"
"""


def setup_instructions(harness: str) -> str:
    """Copy-paste setup for one Full-tier harness (FR-READ-03)."""
    harness = harness.lower()
    command, args = mcp_server()
    if harness == "claude-code":
        return f"""Claude Code: route reads of secret files through safe_read

1. Register the read tool for every project:

   claude mcp add --scope user harpocrates -- {shlex.join([command, *args])}

2. Merge these keys into ~/.claude/settings.json (append to any existing lists and hooks):

{json.dumps(claude_code_settings(), indent=2)}

3. Add this to ~/.claude/CLAUDE.md so the agent knows where to turn when a read is refused:

   {AGENT_NOTE}

4. Check: start `claude` in a repo with a .env and ask it to show the file. The built-in Read is
   refused, and safe_read returns <<HARPO:...>> placeholders instead of the values.

The "hooks" entry scans every file before a built-in Read, a content Grep, or a Bash command such as
`cat` touches it, and refuses the call if the file holds a secret, pointing the agent to safe_read.
Cost: Claude can't Edit a refused file, since Edit needs a built-in Read first (about 3% of ordinary
source files, mostly false alarms). Limit: a script that opens files itself gets through; the egress
gate covers that. For OS-level blocking of the secret paths, see the sandbox step in docs/setup.md."""
    if harness == "codex":
        return f"""Codex CLI: route reads of secret files through safe_read

1. Merge this into ~/.codex/config.toml. `default_permissions` must sit above the first [table]; if you
   already have a profile, add the deny lines to it instead.

{codex_config()}
2. Add this to ~/.codex/AGENTS.md:

   {AGENT_NOTE}

3. Trust the hook: start `codex`, type /hooks, review the Harpocrates PreToolUse hook and trust it.
   Codex runs a new or changed hook only after this review.

4. Check: `codex mcp list` shows harpocrates enabled. In a repo with a .env,
   `codex sandbox -- cat .env` prints "Operation not permitted", and asking Codex to show
   config files that hold a token gets the hook's refusal and a safe_read retry.

Limit: the profile applies to sandboxed commands only. Running with sandbox_mode = "danger-full-access"
or --dangerously-bypass-approvals-and-sandbox turns it off."""
    raise ValueError(f"no setup guide for {harness!r}; choose claude-code or codex")
