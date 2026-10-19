"""Per-harness support tiers (FR-INSTALL-03) and read-tool setup config (FR-READ-03).

The config only denies the harness's built-in reads of secret-bearing paths and registers the
harpocrates MCP server, whose safe_read and safe_grep return those files redacted. Neither harness
applies its read-deny rules to MCP servers, so safe_read can still open what the built-in tools can't.
`harpocrates setup <harness>` prints it; `harpocrates install` writes it into the harness's own files
(FR-INSTALL-01, Harpocrates/install.py).
"""
from __future__ import annotations

import json
import shlex
import shutil
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from Harpocrates.paths import STATE_DIR_NAMES

# File names that hold secrets far more often than not, matched at any depth.
SENSITIVE_FILES = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ecdsa*", "id_ed25519*",
    "*.tfvars", "*.tfstate", "*.jks", "*.keystore", "credentials.json", ".netrc", ".npmrc", ".pypirc",
    ".git-credentials",
)
# Credential stores in the home directory, blocked as a whole.
SENSITIVE_HOME_DIRS = ("~/.ssh", "~/.aws", "~/.azure", "~/.config/gcloud", "~/.kube", "~/.docker", "~/.gnupg")
# The user's own data (the allowlist, the grants, the install record): ~/.harpo, and ~/.harpocrates until it is gone.
# Both are protected from the agent's edits whichever one is in use (FR-NAME-04, paths.py).
STATE_DIRS = tuple(f"~/{name}" for name in STATE_DIR_NAMES)

MCP_TOOLS = ("mcp__harpocrates__safe_read", "mcp__harpocrates__safe_grep")
AGENT_NOTE = (
    "Files that usually hold secrets (.env, keys, credential stores) are blocked for the built-in read "
    "tools. Read them with the harpocrates MCP tools safe_read and safe_grep; secrets come back as "
    "<<HARPO:type:hash>> placeholders. Never try to recover a placeholder's value."
)
# Claude Code asks the user before an edit of such a file (FR-READ-06); Codex has no edit prompt.
CLAUDE_NOTE = AGENT_NOTE + " To edit such a file, use Edit directly: the user is asked once per file and session."
CODEX_NOTE = AGENT_NOTE + " Ask the user to edit such files."
# ponytail: flip when the egress gate ships (M2); until then no harness has more than the read tool.
GATE_AVAILABLE = False


@dataclass(frozen=True)
class Harness:
    """
    Description: Represent a supported coding harness, its detection hints, protection tier, and
        remaining gaps.
    Parameters:
        name: Display name of the harness.
        tier: Advertised support tier.
        commands: Executable names to look for on PATH.
        paths: Config paths under home or application paths under the applications directory.
        gap: Protection limitations displayed after Not protected:.
        setup: Setup guide identifier, or None when no guide exists.
    Output: Immutable dataclass instances providing harness metadata for detection and reports.
    """
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
    """
    Description: Detect harness installations using executables, configuration paths, and
        application directories.
    Parameters:
        home: Home directory to inspect, defaulting to Path.home().
        which: Executable lookup callable, defaulting to shutil.which.
        apps: Applications directory used for .app paths.
    Output: Mapping of harness display names to installation booleans; home lookup, filesystem, and
        lookup callable errors may propagate.
    Notes: Which harnesses look installed: a command on PATH, or a config or app directory.
    """
    home = home or Path.home()

    def present(rel: str) -> bool:
        """
        Description: Check a configured installation path under home or the applications directory.
        Parameters:
            rel: Relative installation path, possibly containing a glob.
        Output: True if the path exists or its glob matches, otherwise False; filesystem lookup
            errors may propagate.
        """
        base = apps if rel.endswith(".app") else home
        return any(base.glob(rel)) if "*" in rel else (base / rel).exists()

    return {h.name: any(which(c) for c in h.commands) or any(present(p) for p in h.paths) for h in HARNESSES}


def tier_report(found: dict[str, bool]) -> str:
    """
    Description: Describe detected harness support and remaining exposure for FR-INSTALL-03.
    Parameters:
        found: Mapping of harness names to detection booleans.
    Output: Human-readable report including setup availability and the unreleased M2 gate
        limitation, or a no-supported-harness message.
    Notes: Support tier per detected harness, with what each tier leaves unprotected
        (FR-INSTALL-03).
    """
    rows = [h for h in HARNESSES if found.get(h.name)]
    if not rows:
        return "No supported harness detected (Claude Code, Codex CLI, OpenCode, Copilot, Cursor)."
    lines = ["Detected harnesses and their Harpocrates support tier:", ""]
    for h in rows:
        lines.append(f"  {h.name}: {h.tier}")
        lines.append(f"    Not protected: {h.gap}")
        if h.setup:
            lines.append(f"    Setup: harpo setup {h.setup}")
        else:
            lines.append("    Protected today: nothing. There is no setup for this harness yet.")
    if not GATE_AVAILABLE:
        lines += ["", "The egress gate isn't released yet (milestone M2). Until it is, every harness, Full tier "
                  "included, is protected by the read tool only: a secret the agent reaches through the "
                  "shell (cat, env, test output, git log -p) still goes to the model."]
    return "\n".join(lines)


def _in_user_site() -> bool:
    """
    Description: Whether this copy of Harpocrates is installed in the user's site-packages (`pip install --user`).
    Parameters: None
    Output: True when this module's file lies under site.getusersitepackages(), otherwise False, also when either path
        cannot be resolved. No side effects.
    """
    try:
        return Path(__file__).resolve().is_relative_to(Path(site.getusersitepackages()).resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _choose_flag(user_site: bool, has_p: bool) -> str:
    """
    Description: Pick the interpreter flags for the generated commands (see python_flag).
    Parameters:
        user_site: whether the package lives in the user site-packages.
        has_p: whether this Python has the -P flag (3.11 and later).
    Output: "-E -P" for a user-site install on a Python with -P, otherwise "-I". No side effects.
    """
    return "-E -P" if user_site and has_p else "-I"


def python_flag() -> str:
    """
    Description: Select the interpreter flags every generated `python -m Harpocrates...` command carries, so neither an
        agent-controlled working directory nor an environment variable can put code in front of Harpocrates (SEC-08).
    Parameters: None
    Output: "-I" (isolated mode), or "-E -P" when the package is installed in the user site-packages and this Python is
        3.11 or later; a string that may hold two flags separated by a space (python_flags() splits it). No process
        is launched.
    Notes: `python -m` puts the current directory first on sys.path, and the current directory is the
        agent-controlled project, so a planted `Harpocrates/` package there would run in place of
        ours (a hook that exits 0 is an allow). PYTHONPATH, PYTHONSTARTUP, PYTHONHOME and the user site
        are the same kind of door, and a project's settings can set environment variables. -I (isolated
        mode: -E -P -s) closes all of them, and is the choice for every install that keeps the package in
        the interpreter's own site-packages. A `pip install --user` install needs the user site, which -I
        drops, so there -E (ignore PYTHON* variables) and -P (3.11+: drop only the cwd entry) are used instead;
        older Pythons have no -P and get -I, so a user-site install is not found there and the hook
        fails closed (exit 2). RESIDUAL RISK of "-E -P": the user site is found from $HOME, so a hook whose
        $HOME an agent controls would load a package planted in that user site. Install into a virtual
        environment (or the interpreter's own site-packages) to avoid it; docs/threat-model.md lists it.
    """
    return _choose_flag(_in_user_site(), sys.version_info >= (3, 11))


def python_flags() -> list[str]:
    """
    Description: The interpreter flags of python_flag() as separate command-line arguments.
    Parameters: None
    Output: ["-I"] or ["-E", "-P"]; the list to put before `-m` when building an argv or a shell command. No process
        is launched.
    """
    return python_flag().split()


def mcp_server() -> tuple[str, list[str]]:
    """
    Description: Build the MCP server invocation using this interpreter for virtual-environment and
        IDE compatibility.
    Parameters: None
    Output: Tuple of interpreter path and argument list with the protective Python flag; does not
        launch the server.
    Notes: This interpreter running the MCP server, so it works from any venv and from IDE-launched
        agents.
    """
    return sys.executable, [*python_flags(), "-m", "Harpocrates.mcp.server"]


def guarded_hook(python: str, *flags: str) -> str:
    """
    Description: Build a PreToolUse hook invocation that maps execution failures to a blocking exit
        status.
    Parameters:
        python: Interpreter path passed safely as shell argument zero.
        *flags: Additional hook flags inserted into the shell script.
    Output: Shell command using sh -c and || exit 2 to fail closed; does not execute it. Post hooks
        must remain bare because they must never block.
    Notes: A PreToolUse hook command that fails closed. Harnesses read only exit 2 as a block, so a
        removed venv (exit 127) or a crash (1) would let the call through. The hook itself exits 0
        or 2, so `|| exit 2` changes nothing else. The explicit `sh -c` works whether the harness
        runs the string in a shell or splits it into argv; the python path travels as $0, so spaces
        and quotes in it are safe. --post hooks stay bare: they must never block.
    """
    script = " ".join(['"$0"', *python_flags(), "-m", "Harpocrates.hooks", *flags, "||", "exit", "2"])
    return shlex.join(["sh", "-c", script, python])


def claude_code_settings() -> dict:
    """
    Description: Build Claude Code permissions and hooks that protect secret reads and
        state-directory edits.
    Parameters: None
    Output: Settings dictionary for ~/.claude/settings.json containing root-anchored denies, allowed
        MCP tools, a guarded asking PreToolUse hook, and bare post/session hooks; writes nothing.
    Notes: Lines for ~/.claude/settings.json. `//**/` anchors at the filesystem root, so a rule in
        user settings covers every project and any file outside the working directory.
    """
    deny = [f"Read(//**/{name})" for name in SENSITIVE_FILES] + [f"Read({d}/**)" for d in SENSITIVE_HOME_DIRS]
    # only the user edits the allowlist; Edit rules cover Write too
    deny.extend(f"Edit({d}/**)" for d in STATE_DIRS)
    command, _ = mcp_server()
    # Content-aware: also blocks built-in reads of any other file that holds a secret (Harpocrates/hooks.py).
    # --ask: an edit of such a file prompts the user instead of being refused (reads are always refused, FR-READ-06).
    hook = {"type": "command", "command": guarded_hook(command, "--ask")}
    # --post records a Yes: PostToolUse/PostToolUseFailure fire only for an edit that ran, SessionEnd drops the grants.
    post = {"type": "command", "command": shlex.join([command, *python_flags(), "-m", "Harpocrates.hooks", "--post"])}
    edits = "Edit|MultiEdit|Write|NotebookEdit"
    return {"permissions": {"deny": deny, "allow": list(MCP_TOOLS)},
            "hooks": {"PreToolUse": [{"matcher": f"Read|Grep|Bash|{edits}", "hooks": [hook]}],
                      "PostToolUse": [{"matcher": edits, "hooks": [post]}],
                      "PostToolUseFailure": [{"matcher": edits, "hooks": [post]}],
                      "SessionEnd": [{"hooks": [post]}]}}


def codex_config_parts() -> tuple[str, str]:
    """
    Description: Build Codex MCP, sandbox, and content-aware hook configuration as separate root and
        table sections.
    Parameters: None
    Output: Tuple of TOML root settings and tables for ~/.codex/config.toml; writes nothing.
        Profiles limit sandboxed shell commands but do not restrict MCP reads through safe_read.
    Notes: Lines for ~/.codex/config.toml. The profile limits sandboxed shell commands; Codex
        doesn't apply profiles to MCP servers, so safe_read still reads these paths. The PreToolUse
        hook is the same content-aware check as Claude Code's: it refuses shell commands that would
        print a file holding a secret.
    """
    command, args = mcp_server()
    files = "\n".join(f'"**/{name}" = "deny"' for name in SENSITIVE_FILES)
    dirs = "\n".join(f'"{d}" = "deny"' for d in (*SENSITIVE_HOME_DIRS, *STATE_DIRS))
    root = 'default_permissions = "harpocrates"\n'
    tables = f"""[mcp_servers.harpocrates]
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

# Deny-only: Codex parses a PreToolUse "ask" but doesn't support it and fails open, so this hook runs without ask mode.
[[hooks.PreToolUse]]
matcher = "^Bash$"

[[hooks.PreToolUse.hooks]]
type = "command"
command = {json.dumps(guarded_hook(command), ensure_ascii=False)}
statusMessage = "Harpocrates: checking files for secrets"
"""
    return root, tables


def codex_config() -> str:
    """
    Description: Combine Codex root settings and tables for the setup command.
    Parameters: None
    Output: Complete configuration string; writes no files.
    Notes: The same lines in one piece, for `harpocrates setup codex`.
    """
    root, tables = codex_config_parts()
    return f"{root}\n{tables}"


def setup_instructions(harness: str) -> str:
    """
    Description: Generate copy-paste read-tool setup instructions for a Full-tier harness under
        FR-READ-03.
    Parameters:
        harness: Case-insensitive guide identifier: claude-code or codex.
    Output: Setup guide including validation steps and protection limitations; raises ValueError for
        unsupported harnesses and writes nothing.
    Notes: Copy-paste setup for one Full-tier harness (FR-READ-03).
    """
    harness = harness.lower()
    command, args = mcp_server()
    if harness == "claude-code":
        return f"""Claude Code: route reads of secret files through safe_read

1. Register the read tool for every project:

   claude mcp add --scope user harpocrates -- {shlex.join([command, *args])}

2. Merge these keys into ~/.claude/settings.json (append to any existing lists and hooks):

{json.dumps(claude_code_settings(), indent=2)}

3. Add this to ~/.claude/CLAUDE.md so the agent knows where to turn when a read is refused:

   {CLAUDE_NOTE}

4. Check: start `claude` in a repo with a .env and ask it to show the file. The built-in Read is
   refused, and safe_read returns <<HARPO:...>> placeholders instead of the values.

The "hooks" entry scans every file before a built-in Read, Edit or Write, a content Grep, or a Bash
command such as `cat` touches it. If the file holds a secret, a read is refused with a pointer to
safe_read / safe_grep, in every mode. Only an EDIT of such a file prompts you ("Harpocrates:" naming the
file), once per file and session, even in accept-edits mode; Yes lets Claude change it (and read it
as-is, so its values reach the provider), No leaves it untouched. A folder of known false alarms (hashes,
build output) can be allowed for good with `harpo allow <path>`; each pass is logged in
~/.harpo/allow.log. The deny rules above still win over the allowlist.
Limit: a script that opens files itself gets through; the egress
gate covers that. For OS-level blocking of the secret paths, see the sandbox step in docs/setup.md."""
    if harness == "codex":
        return f"""Codex CLI: route reads of secret files through safe_read

1. Merge this into ~/.codex/config.toml. `default_permissions` must sit above the first [table]; if you
   already have a profile, add the deny lines to it instead.

{codex_config()}
2. Add this to ~/.codex/AGENTS.md:

   {CODEX_NOTE}

3. Trust the hook: start `codex`, type /hooks, review the Harpocrates PreToolUse hook and trust it.
   Codex runs a new or changed hook only after this review.

4. Check: `codex mcp list` shows harpocrates enabled. In a repo with a .env,
   `codex sandbox -- cat .env` prints "Operation not permitted", and asking Codex to show
   config files that hold a token gets the hook's refusal and a safe_read retry.

Limit: the profile applies to sandboxed commands only. Running with sandbox_mode = "danger-full-access"
or --dangerously-bypass-approvals-and-sandbox turns it off."""
    raise ValueError(f"no setup guide for {harness!r}; choose claude-code or codex")
