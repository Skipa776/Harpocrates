"""SEC-10: a repo cannot route Claude Code around the gate.

When the install record says the gate is on, the PreToolUse hook (hooks.py) asks this module whether the settings
Claude Code will use send its requests to the gate. If anything sends them elsewhere, every tool call is refused with
a plain deny that names the setting file. Nothing waits for an answer: no prompt, no ask, in any mode. With no gate
installed, none of this runs and nothing changes. `harpo doctor` (G7, FR-INSTALL-02) will call bypass_findings and
fail on each finding; that command does not exist yet.

The install record. ~/.harpo/install.json (paths.state_dir) holds, next to the per-harness sections install writes,
one more top-level entry, written by `harpo guard up` (G7) and read here:

    "gate": {"enabled": true, "port": 47321, "token_header": "x-harpo-gate-token"}

  * absent entry, absent record, or "enabled": false  -> no gate installed; this module reports nothing.
  * "enabled": true needs "port" (an integer, 1 to 65535: where http://127.0.0.1:<port> listens) and "token_header"
    (the HTTP header name that carries the gate token, see docs/gate-proxy-spec.md). Anything else about the entry is
    an error and the hook refuses (fail closed): without it Harpo cannot tell whether the gate is on. So is a record
    that exists but cannot be read (permissions, a folder, over 1 MiB), and a record that is not strict JSON when its
    bytes contain the quoted word "gate" (a gate entry may be there, damaged). A record that is not strict JSON and
    does not contain it, or is JSON but not an object, cannot hold a gate entry and counts as no gate; the existing
    state-folder tests keep junk in install.json and expect the hook not to change. A ~/.harpo that is a symlink or a
    file reads as no record, as every other reader does (SEC-09). Unknown keys are ignored.

What counts as a bypass. For each of ANTHROPIC_BASE_URL, CLAUDE_CODE_USE_BEDROCK, CLAUDE_CODE_USE_VERTEX and
ANTHROPIC_CUSTOM_HEADERS the effective value is the one from the highest-precedence source that sets it, even to an
empty string (a settings file cannot remove a variable, so an empty value is how it overrides one). It is a bypass when
  * ANTHROPIC_BASE_URL is not http://127.0.0.1:<port> (localhost and [::1] are the same host; the port must be given
    and match; the path, query, fragment and user info must be absent; matched as one strict pattern, not a URL parser),
    or is not set at all (Claude Code then uses its default, the provider);
  * CLAUDE_CODE_USE_BEDROCK or CLAUDE_CODE_USE_VERTEX is on (anything but empty, 0, false, no, off);
  * ANTHROPIC_CUSTOM_HEADERS does not carry the token header with a value (not set counts as lacking it).

Precedence, highest first (https://code.claude.com/docs/en/settings, "Settings precedence", checked 2026-10-19):
managed settings, command line, <repo>/.claude/settings.local.json, <repo>/.claude/settings.json,
~/.claude/settings.json. An `env` entry is written into the process environment and replaces the value inherited from
the shell (https://code.claude.com/docs/en/env-vars, "Precedence", same date), so the hook's own environment ranks
last. G2 saw the same (docs/gate-transport.md, T2): a project value beat both a user value and an exported variable.
Managed settings read here are the files: managed-settings.json, then managed-settings.d/*.json in name order, a later
file winning (https://code.claude.com/docs/en/managed-settings, "Where each mechanism stores the policy"), in
/Library/Application Support/ClaudeCode, /etc/claude-code or C:\\Program Files\\ClaudeCode. Not read: MDM profiles, the
Windows registry, server-managed settings and `--settings`; Claude Code copies what they set into the process
environment, which this module does read. Settings are looked up in the event's cwd and in CLAUDE_PROJECT_DIR, since a
`cd` can move the cwd away from where Claude Code started; a bypass in either refuses.

Fail closed. A settings file that exists but cannot be read, is not a regular file, is over 1 MiB, or is not strict
JSON (duplicate keys, NaN, a BOM, trailing text, a non-text value for one of the four variables) is itself a finding, and
then only such findings are reported: the effective values are unknown. Files are opened without blocking, so a pipe
planted in a repo cannot hang the hook (a hook that times out is treated as an allow). Messages carry file names, line
numbers and the host:port of a wrong base URL, never a path, query, user info, header or token.

Codex. Nothing here reads a project .codex file. Codex ignores model_provider, model_providers and openai_base_url in a
project config (user-level only; docs/gate-transport.md, source X1), so a repo cannot move Codex off the gate that way.
The hook skips this check for Codex events, which carry `turn_id` and `model` (documented as Codex-specific fields,
https://learn.chatgpt.com/docs/hooks, checked 2026-10-19): Codex does not read Claude Code settings.
"""
from __future__ import annotations

import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from Harpocrates import paths

WATCHED = ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_CUSTOM_HEADERS")
MAX_BYTES = 1 << 20  # a settings file or the record is a few hundred bytes; more than this is not one
RECORD = "install.json"

_GATE_URL = re.compile(r"http://(localhost|127\.0\.0\.1|\[::1\]):([0-9]{1,5})", re.IGNORECASE)
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # an HTTP token (RFC 9110, 5.6.2)
_TARGET = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{0,15}://(?:[^/?#@\\\s]*@)?(\[[0-9A-Fa-f:.]{2,45}\]|[A-Za-z0-9.-]{1,253})"
                     r"(?::([0-9]{1,5}))?(?=[/?#]|$)")
_NOT_JSON = ("not valid JSON", "not valid UTF-8")  # the _Doc problems that leave bytes to look at
_OFF = {"", "0", "false", "no", "off"}  # what switches a provider flag off; anything else is on
_PROVIDERS = (("CLAUDE_CODE_USE_BEDROCK", "Amazon Bedrock"), ("CLAUDE_CODE_USE_VERTEX", "Google Vertex AI"))
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
ENVIRONMENT = "the environment Claude Code runs hooks in"


@dataclass(frozen=True)
class Gate:
    """
    Description: The gate as the install record describes it when it is on.
    Parameters:
        port: The port of the gate on 127.0.0.1.
        token_header: The HTTP header name that carries the gate token.
    Output: An immutable value; url gives the address Claude Code must use.
    """

    port: int
    token_header: str

    @property
    def url(self) -> str:
        """
        Description: The one address Claude Code's base URL must be, in its canonical spelling.
        Parameters:
            None: computed from the port.
        Output: http://127.0.0.1:<port>, with no path.
        """
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class Finding:
    """
    Description: One reason Claude Code's requests would not reach the gate, ready to show to a person.
    Parameters:
        where: The file (or the environment) that holds the setting.
        setting: The variable name, or "gate" for a problem with the install record.
        text: One plain sentence that names the file, the setting and what is wrong; it holds no secret value.
    Output: An immutable value that `harpo doctor` prints and the hook joins into its refusal.
    """

    where: str
    setting: str
    text: str


class GateRecordError(ValueError):
    """
    Description: The install record cannot be read, or its gate entry is malformed, so Harpo cannot tell whether the
        gate is on.
    Parameters:
        label: The record's name as shown to the user, for example ~/.harpo/install.json.
        reason: A short fixed phrase, never file content.
    Output: Raised by load_gate; str() is the plain sentence; the label is kept as .label.
    """

    def __init__(self, label: str, reason: str) -> None:
        """
        Description: Build the message from the record's name and the reason.
        Parameters:
            label: The record's name as shown to the user.
            reason: A short fixed phrase.
        Output: None. Sets .label and the exception message.
        """
        self.label = label
        super().__init__(f"{label} cannot be read as the Harpo install record ({reason}), so Harpo cannot tell "
                         "whether the gate is on. Fix or delete its gate entry.")


@dataclass(frozen=True)
class _Doc:
    """
    Description: A JSON file as read: absent, a problem, or parsed.
    Parameters:
        present: Whether the file exists.
        data: The parsed JSON, when there is no problem.
        text: The file's text, for finding line numbers.
        raw: The file's bytes when it is not UTF-8 or not strict JSON, so the record reader can look for a gate entry.
        problem: A short fixed phrase when it cannot be used, else empty.
    Output: An immutable value.
    """

    present: bool = False
    data: object = None
    text: str = ""
    raw: bytes = b""
    problem: str = ""


@dataclass(frozen=True)
class _Source:
    """
    Description: One place that can set the four variables.
    Parameters:
        label: How a message names it.
        values: The watched variables it sets, as text.
        lines: The line each is on, when known.
    Output: An immutable value.
    """

    label: str
    values: Mapping[str, str]
    lines: Mapping[str, int]


def managed_dir() -> Path:
    """
    Description: The folder where an administrator puts managed-settings.json on this operating system
        (https://code.claude.com/docs/en/managed-settings, checked 2026-10-19).
    Parameters:
        None: chosen by platform.
    Output: /Library/Application Support/ClaudeCode on macOS, C:\\Program Files\\ClaudeCode on Windows, else
        /etc/claude-code. Nothing is read; tests replace this function.
    """
    if sys.platform == "darwin":
        return Path("/Library/Application Support/ClaudeCode")
    if sys.platform.startswith("win"):
        return Path("C:\\Program Files\\ClaudeCode")
    return Path("/etc/claude-code")


def gate_url_matches(value: str, gate: Gate) -> bool:
    """
    Description: Whether a base URL is the gate: http, host localhost or 127.0.0.1 or [::1], the gate's port, and nothing
        after the port. One strict pattern over the whole string, so no URL parser can read it differently from
        Claude Code (user info, backslashes, spaces, a trailing slash, a query, https and a missing port all fail).
    Parameters:
        value: The base URL text from a setting.
        gate: The gate from the install record.
    Output: True only for an exact match; False for everything else. No side effects.
    """
    found = _GATE_URL.fullmatch(value)
    return found is not None and int(found.group(2)) == gate.port


def _no_constant(_name: str) -> None:
    """
    Description: Refuse NaN and Infinity, which are not JSON.
    Parameters:
        _name: The constant the parser met.
    Output: Never returns; raises ValueError.
    """
    raise ValueError("non-standard number")


def _no_duplicates(pairs: list) -> dict:
    """
    Description: Build an object from its pairs and refuse a repeated key, which parsers resolve differently.
    Parameters:
        pairs: The (key, value) pairs of one JSON object, in file order.
    Output: The dictionary; raises ValueError on a repeated key.
    """
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def _load(path: Path) -> _Doc:
    """
    Description: Read and strictly parse one JSON file without ever blocking or following a mistake into a loop.
    Parameters:
        path: The file; a symlink is followed, as Claude Code does.
    Output: An absent _Doc when the file or a folder above it is missing; a _Doc with a problem phrase when it is not a
        regular file, is over MAX_BYTES, cannot be opened or read, is not UTF-8, or is not strict JSON (duplicate keys,
        NaN, a BOM or trailing text); else a _Doc with the data. Never raises for a file problem.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))  # a pipe must not block the open
    except (FileNotFoundError, NotADirectoryError):
        return _Doc()
    except OSError:
        return _Doc(present=True, problem="unreadable")
    try:
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return _Doc(present=True, problem="not a regular file")
            raw = handle.read(MAX_BYTES + 1)
    except OSError:
        return _Doc(present=True, problem="unreadable")
    if len(raw) > MAX_BYTES:
        return _Doc(present=True, problem="too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _Doc(present=True, raw=raw, problem="not valid UTF-8")
    try:
        data = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_no_duplicates)
    except (ValueError, RecursionError):  # JSONDecodeError, duplicate keys, NaN, a huge integer, nesting too deep
        return _Doc(present=True, text=text, raw=raw, problem="not valid JSON")
    return _Doc(present=True, data=data, text=text)


def load_gate() -> Optional[Gate]:
    """
    Description: Read the gate entry of the install record.
    Parameters:
        None: the record is paths.peek_state_dir() / install.json, looked up on every call, with no side effects.
    Output: The Gate when the record says the gate is on; None when there is no gate (no record, a state folder that
        is a symlink or a file, no gate entry, or "enabled": false). Raises GateRecordError when the record or its
        gate entry cannot be used (see the module docstring); only paths.home can raise anything else.
    """
    state = paths.peek_state_dir()
    if paths.unusable(state):
        return None
    label = f"~/{state.name}/{RECORD}"
    doc = _load(state / RECORD)
    if doc.problem in _NOT_JSON:
        # A record that is not strict JSON: nothing says a gate was ever installed unless the file names one. Refuse
        # only then (a truncated or half-edited gate entry); other junk predates the gate and changes nothing.
        if b'"gate"' in doc.raw:
            raise GateRecordError(label, doc.problem)
        return None
    if doc.problem:  # present but not readable at all (permissions, a folder, over the size cap): cannot tell
        raise GateRecordError(label, doc.problem)
    if not doc.present or not isinstance(doc.data, dict) or "gate" not in doc.data:
        return None  # an array, a string or a number cannot hold a gate entry
    entry = doc.data["gate"]
    if not isinstance(entry, dict):
        raise GateRecordError(label, "gate is not an object")
    enabled = entry.get("enabled")
    if not isinstance(enabled, bool):
        raise GateRecordError(label, "gate.enabled is not true or false")
    if not enabled:
        return None
    port, header = entry.get("port"), entry.get("token_header")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise GateRecordError(label, "gate.port is not a port number")
    if not isinstance(header, str) or not _HEADER_NAME.fullmatch(header):
        raise GateRecordError(label, "gate.token_header is not an HTTP header name")
    return Gate(port=port, token_header=header)


def _line_of(text: str, name: str) -> Optional[int]:
    """
    Description: The line a settings key is written on (the last one, which is the one a parser keeps).
    Parameters:
        text: The file's text.
        name: The key.
    Output: The 1-based line number, or None when the key is not written in a plain way.
    """
    pattern = re.compile(r'"' + re.escape(name) + r'"\s*:')
    found = None
    for number, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            found = number
    return found


def _watched(doc: _Doc) -> tuple[dict[str, str], dict[str, int], str]:
    """
    Description: Take the four variables out of a settings document's `env` block.
    Parameters:
        doc: A present, parsed settings file.
    Output: (values as text, their lines, problem phrase). A JSON number or boolean becomes its JSON text; a document
        that is not an object, an `env` that is not an object, or a watched value that is not text, a number or a
        boolean gives a problem and no values. Other keys of `env` are not looked at.
    """
    if not isinstance(doc.data, dict):
        return {}, {}, "not a JSON object"
    if "env" not in doc.data:
        return {}, {}, ""
    env = doc.data["env"]
    if not isinstance(env, dict):
        return {}, {}, "env is not an object"
    values: dict[str, str] = {}
    for name in WATCHED:
        if name not in env:
            continue
        value = env[name]
        if isinstance(value, str):
            values[name] = value
        elif isinstance(value, bool):
            values[name] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            values[name] = json.dumps(value)
        else:
            return {}, {}, f"{name} is not text"
    lines = {name: n for name in values if (n := _line_of(doc.text, name)) is not None}
    return values, lines, ""


def _plain(text: str) -> str:
    """
    Description: Make a path safe to put in a one-line message.
    Parameters:
        text: Any text.
    Output: The text with control characters (a newline in a folder name) replaced by a question mark.
    """
    return _CONTROL.sub("?", text)


def _settings_files(project: Path, home: Path, managed: Path) -> tuple[list[tuple[str, Path]], list[Finding]]:
    """
    Description: List the settings files that can set the variables for one project, highest precedence first.
    Parameters:
        project: The folder Claude Code was started in (or the event's cwd).
        home: The user's home folder (paths.home).
        managed: The folder of managed-settings.json (managed_dir).
    Output: ((label, path) pairs: managed-settings.d files last-name-first, managed-settings.json, the project's
        settings.local.json and settings.json, ~/.claude/settings.json) and findings for a managed-settings.d folder
        that exists but cannot be listed. Nothing is opened.
    """
    found: list[Finding] = []
    files: list[tuple[str, Path]] = []
    dropin = managed / "managed-settings.d"
    try:
        with os.scandir(dropin) as scan:
            names = sorted(e.name for e in scan if e.name.endswith(".json") and not e.name.startswith("."))
    except (FileNotFoundError, NotADirectoryError):
        names = []
    except OSError:
        names = []
        label = _plain(str(dropin))
        found.append(Finding(label, "managed-settings.d",
                             f"{label} cannot be read as a managed settings folder (unreadable), so Harpo cannot "
                             "tell where Claude Code sends requests."))
    files += [(_plain(str(dropin / n)), dropin / n) for n in reversed(names)]
    files.append((_plain(str(managed / "managed-settings.json")), managed / "managed-settings.json"))
    for name in ("settings.local.json", "settings.json"):
        files.append((_plain(str(project / ".claude" / name)), project / ".claude" / name))
    files.append(("~/.claude/settings.json", home / ".claude" / "settings.json"))
    return files, found


def _sources(project: Path, home: Path, managed: Path) -> tuple[list[_Source], list[Finding]]:
    """
    Description: Read every settings file that can apply to a project.
    Parameters:
        project: The folder Claude Code was started in (or the event's cwd).
        home: The user's home folder.
        managed: The folder of managed-settings.json.
    Output: (sources with the watched values they set, highest precedence first; findings for each file that exists but
        cannot be used). A missing file is neither.
    """
    files, found = _settings_files(project, home, managed)
    sources: list[_Source] = []
    for label, path in files:
        doc = _load(path)
        problem = doc.problem
        values: dict[str, str] = {}
        lines: dict[str, int] = {}
        if doc.present and not problem:
            values, lines, problem = _watched(doc)
        if problem:
            found.append(Finding(label, "settings", f"{label} cannot be read as Claude Code settings ({problem}), so "
                                 "Harpo cannot tell where Claude Code sends requests."))
        elif values:
            sources.append(_Source(label, values, lines))
    return sources, found


def _effective(name: str, sources: list[_Source], environ: Mapping[str, str]) -> tuple[Optional[str], Optional[_Source]]:
    """
    Description: The value Claude Code ends up with for one variable.
    Parameters:
        name: One of WATCHED.
        sources: The settings sources, highest precedence first.
        environ: The hook's own environment, which ranks below every settings file.
    Output: (value, the source that set it); (value, None) when only the environment sets it; (None, None) when
        nothing does. An empty string counts as set.
    """
    for source in sources:
        if name in source.values:
            return source.values[name], source
    if name in environ:
        return environ[name], None
    return None, None


def _where(source: Optional[_Source], name: str) -> str:
    """
    Description: How a message names the place a setting came from.
    Parameters:
        source: The settings source, or None for the environment.
        name: The variable, for the line number.
    Output: The file name with " (line N)" when known, or the environment's name.
    """
    if source is None:
        return ENVIRONMENT
    line = source.lines.get(name)
    return source.label + (f" (line {line})" if line else "")


def _target(value: str) -> Optional[str]:
    """
    Description: The host and port of a URL, for a message, and nothing else of it.
    Parameters:
        value: A base URL text from a setting.
    Output: "host" or "host:port" in a safe character set, or None when the text is not an address of that shape.
        User info, path, query and fragment are never returned.
    """
    found = _TARGET.match(value.strip())
    if found is None:
        return None
    host, port = found.groups()
    return f"{host}:{port}" if port else host


def _has_token(value: str, header: str) -> bool:
    """
    Description: Whether an ANTHROPIC_CUSTOM_HEADERS value (Name: Value lines) carries the token header.
    Parameters:
        value: The setting's text.
        header: The token header name from the install record; names compare without regard to case.
    Output: True when a line has that name before a colon and a non-empty value after it; else False. The value is
        never returned or logged.
    """
    for line in value.splitlines():
        name, colon, rest = line.partition(":")
        if colon and name.strip().lower() == header.lower() and rest.strip():
            return True
    return False


def _judge(gate: Gate, sources: list[_Source], environ: Mapping[str, str]) -> list[Finding]:
    """
    Description: Check the four variables' effective values against the gate.
    Parameters:
        gate: The gate from the install record.
        sources: The settings sources for one project, highest precedence first.
        environ: The hook's own environment.
    Output: One Finding per variable that sends requests around the gate, in the order of WATCHED; empty when none does.
    """
    found: list[Finding] = []
    value, source = _effective("ANTHROPIC_BASE_URL", sources, environ)
    where = _where(source, "ANTHROPIC_BASE_URL")
    if value is None:
        found.append(Finding(ENVIRONMENT, "ANTHROPIC_BASE_URL",
                             "ANTHROPIC_BASE_URL is not set in any Claude Code settings file or in the environment, so "
                             f"Claude Code would send requests straight to the provider; set it to {gate.url} in "
                             "~/.claude/settings.json."))
    elif not gate_url_matches(value, gate):
        if not value.strip():
            text = (f"{where} sets ANTHROPIC_BASE_URL to an empty value, so Claude Code uses the provider directly; "
                    f"the gate address is {gate.url}.")
        else:
            text = (f"{where} sets ANTHROPIC_BASE_URL to {_target(value) or 'a value that is not an address'}, "
                    f"which is not the gate address {gate.url}.")
        found.append(Finding(where, "ANTHROPIC_BASE_URL", text))
    for name, label in _PROVIDERS:
        value, source = _effective(name, sources, environ)
        if value is not None and value.strip().lower() not in _OFF:
            where = _where(source, name)
            found.append(Finding(where, name, f"{where} turns on {name}, which sends requests to {label} instead of "
                                 "the gate."))
    value, source = _effective("ANTHROPIC_CUSTOM_HEADERS", sources, environ)
    if value is None:
        found.append(Finding(ENVIRONMENT, "ANTHROPIC_CUSTOM_HEADERS",
                             "ANTHROPIC_CUSTOM_HEADERS is not set in any Claude Code settings file or in the "
                             f"environment, so Claude Code does not send the gate token header ({gate.token_header})."))
    elif not _has_token(value, gate.token_header):
        where = _where(source, "ANTHROPIC_CUSTOM_HEADERS")
        found.append(Finding(where, "ANTHROPIC_CUSTOM_HEADERS",
                             f"{where} sets ANTHROPIC_CUSTOM_HEADERS without the gate token header "
                             f"({gate.token_header})."))
    return found


def _roots(cwd: Path, environ: Mapping[str, str]) -> list[Path]:
    """
    Description: The folders whose .claude settings can apply to this session.
    Parameters:
        cwd: The event's working directory.
        environ: The hook's environment, for CLAUDE_PROJECT_DIR.
    Output: The cwd, then CLAUDE_PROJECT_DIR when it is an absolute path that differs (a `cd` moves the cwd away from
        where Claude Code started). Only ever adds a folder to check, so it can only make the check stricter.
    """
    roots = [cwd]
    started = environ.get("CLAUDE_PROJECT_DIR")
    if started and os.path.isabs(started) and Path(started) != cwd:
        roots.append(Path(started))
    return roots


def bypass_findings(project_dir: Path, environ: Optional[Mapping[str, str]] = None) -> list[Finding]:
    """
    Description: List every way Claude Code would send its requests around the gate, for the hook and for
        `harpo doctor` (G7), which does not exist yet and will fail on each finding this returns.
    Parameters:
        project_dir: The project folder (the hook passes the event's cwd).
        environ: The environment to judge; defaults to os.environ.
    Output: An empty list when no gate is installed (see load_gate) or nothing bypasses it. Otherwise one Finding per
        setting that bypasses it, naming the file, or, when the install record or a settings file cannot be used,
        only the findings for those. Reads the record and the settings files; writes nothing. Does not raise for bad
        files; paths.home can still raise RuntimeError.
    """
    env = os.environ if environ is None else environ
    try:
        gate = load_gate()
    except GateRecordError as exc:
        return [Finding(exc.label, "gate", str(exc))]
    if gate is None:
        return []
    home, managed = paths.home(), managed_dir()
    problems: dict[str, Finding] = {}
    judged: dict[str, Finding] = {}
    for root in _roots(project_dir, env):
        sources, bad = _sources(root, home, managed)
        problems.update({f.text: f for f in bad})
        if not bad:
            judged.update({f.text: f for f in _judge(gate, sources, env)})
    return list((problems or judged).values())


def refusal(cwd: Path, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """
    Description: The hook's decision: the text of the refusal when a tool call must be denied for SEC-10.
    Parameters:
        cwd: The event's working directory.
        environ: The environment to judge; defaults to os.environ.
    Output: None when there is no gate or no bypass (the hook goes on as before); otherwise one plain line that names
        each setting file and says nothing is waiting for an answer. The caller denies with it, in every mode.
    """
    found = bypass_findings(cwd, environ)
    if not found:
        return None
    return ("Harpo gate bypass: Claude Code would send this session's requests around the Harpo gate, so every tool "
            "call is refused. " + " ".join(f.text for f in found) + " Fix this, then restart the session if it "
            "persists. Nothing is waiting for an answer.")
