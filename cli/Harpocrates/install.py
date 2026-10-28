"""FR-INSTALL-01: write the read-tool setup into each harness's own files, and take it out again.

Claude Code: ~/.claude/settings.json (deny/allow rules and hooks, merged), the harpo MCP server (through
`claude mcp`), and a marked note in ~/.claude/CLAUDE.md. Codex: a marked block in ~/.codex/config.toml and a
marked note in ~/.codex/AGENTS.md. Every file is planned in memory first, so a refusal (unparsable file, a clash
with the user's own config) writes nothing for that harness. Writes are atomic, keep the file mode, and the first
original is kept once as <file>.harpo.bak (FR-INSTALL-08). Nothing here talks to the network.

FR-NAME-01 migration: before the rename the MCP server, the allow rules, the marked blocks, the Codex profile and the
backups were all named `harpocrates`. A new install moves what it owns over to `harpo` (install record for the rules and
the registration, the old marks for the blocks) and never drops a deny rule; uninstall removes both the new and the old
names. Anything the user wrote under the old names themselves is left alone.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from Harpocrates import harness, paths
from Harpocrates.paths import state_dir

try:  # stdlib from 3.11; the package supports 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

TARGETS = ("claude-code", "codex")
NAMES = {"claude-code": "Claude Code", "codex": "Codex CLI"}
BACKUP_SUFFIX = ".harpo.bak"
LEGACY_BACKUP_SUFFIX = ".harpocrates.bak"  # an install from before the rename kept its first original under this name
HOOK_MARK = "-m Harpocrates.hooks"  # in the command of every hook entry we write, whatever the python path
MCP_NAME = harness.MCP_NAME
LEGACY_MCP_NAME = harness.LEGACY_MCP_NAME
HOOKS_TRUST_NOTE = (
    "One manual step: Codex runs a new or changed hook only after you trust it. Start `codex`, type /hooks, "
    "review the Harpo PreToolUse hook and trust it (once)."
)


NO_RECORD_NOTE = (
    "No install record (~/.harpo/install.json): removed only Harpo's marked blocks and hook commands. "
    "Permission rules and the MCP server are left, since they may be yours; drop them by hand if not."
)


class RefusalError(Exception):
    """
    Description: Represent an installation that cannot proceed safely, with a reason and corrective
        action.
    Parameters:
        *args: Exception arguments forwarded to Exception.
    Output: Exception instances reporting a refusal before planned files are written.
    """


@dataclass(frozen=True)
class Step:
    """
    Description: Represent one immutable installation or removal result.
    Parameters:
        label: Display label for the affected component.
        state: Result: added, updated, removed, unchanged, manual, or failed.
        detail: Additional explanatory text, empty by default.
    Output: Instances providing a component label, status, and detail for reports.
    """
    label: str
    state: str  # added | updated | removed | unchanged | manual | failed
    detail: str = ""


@dataclass
class Report:
    """
    Description: Collect the steps, notes, and refusal for a harness operation.
    Parameters:
        target: Harness setup identifier.
        steps: Operation results, defaulting to a new empty list.
        notes: User-facing notes, defaulting to a new empty list.
        refused: Refusal explanation, or None.
    Output: Mutable instances used to accumulate installation or removal outcomes.
    """
    target: str
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    refused: Optional[str] = None


@dataclass(frozen=True)
class _Marks:
    """
    Description: Represent the paired delimiters surrounding an owned configuration block.
    Parameters:
        start: Opening marker text.
        end: Closing marker text.
    Output: Immutable instances providing start and end markers.
    """
    start: str
    end: str


MD_MARKS = _Marks("<!-- harpo:start -->", "<!-- harpo:end -->")
TOML_MARKS = _Marks("# >>> harpo >>>", "# <<< harpo <<<")
LEGACY_MD_MARKS = _Marks("<!-- harpocrates:start -->", "<!-- harpocrates:end -->")  # blocks from before the rename
LEGACY_TOML_MARKS = _Marks("# >>> harpocrates >>>", "# <<< harpocrates <<<")


@dataclass(frozen=True)
class _Edit:
    """
    Description: Represent a planned file change before it is applied.
    Parameters:
        label: Display name of the edited component.
        path: File to edit.
        old: Original text, or None for a nonexistent file.
        new: Replacement text, or None to delete the file.
        remove: Whether the operation is removal.
        private: Whether this is private state requiring a private new directory and no backup.
    Output: Immutable instances providing change, state, backup-path, and backup-needed properties.
    """
    label: str
    path: Path
    old: Optional[str]  # None: the file doesn't exist
    new: Optional[str]  # None: delete the file
    remove: bool
    private: bool = False  # the install record: dir 0700, no backup copy
    backup_first: bool = True  # False: never copy the current text to <file>.harpo.bak (it may hold the gate token)

    @property
    def changed(self) -> bool:
        """
        Description: Compare the planned and original text to identify a needed change.
        Parameters: None
        Output: True if new and old differ, otherwise False.
        """
        return self.new != self.old

    @property
    def state(self) -> str:
        """
        Description: Classify a planned edit for reporting.
        Parameters: None
        Output: One of unchanged, removed, added, or updated, based on content and operation type.
        """
        if not self.changed:
            return "unchanged"
        return "removed" if self.remove else ("added" if self.old is None else "updated")

    @property
    def backup(self) -> Path:
        """
        Description: Derive the adjacent backup path for the original file.
        Parameters: None
        Output: Path with BACKUP_SUFFIX appended to its name; creates nothing.
        """
        return self.path.with_name(self.path.name + BACKUP_SUFFIX)

    @property
    def needs_backup(self) -> bool:
        """
        Description: Check whether a changed nonprivate existing file needs its first backup.
        Parameters: None
        Output: True when no backup exists (under the new or the pre-rename name) and the edit qualifies, otherwise False; filesystem
            lookup errors may propagate.
        """
        legacy = self.path.with_name(self.path.name + LEGACY_BACKUP_SUFFIX)  # that first original already exists
        return (self.changed and self.old is not None and not self.private and self.backup_first
                and not self.backup.exists() and not legacy.exists())


# --- reading and writing files -------------------------------------------------------------------------------------

def _read(path: Path) -> Optional[str]:
    """
    Description: Read UTF-8 file bytes while preserving line endings for safe configuration
        planning.
    Parameters:
        path: File to read.
    Output: Text preserving CRLF and missing final newline, or None for a missing file; raises
        RefusalError for other read or decoding failures.
    """
    try:
        return path.read_bytes().decode("utf-8")  # bytes: keeps CRLF and a missing final newline as they are
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusalError(f"Can't read {_tilde(path)} ({exc}); nothing was written.") from exc


def _write(edit: _Edit) -> None:
    """
    Description: Apply a planned edit with a first-original backup and atomic replacement of the
        resolved target.
    Parameters:
        edit: Planned file change to apply.
    Output: None; may back up, delete, create directories, fsync, chmod, and replace files. Keeps
        existing modes, uses 0o600 for new files, and removes an empty private directory on
        deletion. Cleans temporary files on failure and propagates errors.
    """
    if edit.needs_backup:
        shutil.copy2(edit.path, edit.backup)
    if edit.new is None:
        edit.path.unlink()
        if edit.private:
            try:
                edit.path.parent.rmdir()  # only if we left it empty
            except OSError:
                pass
        return
    # Trade-off: a dotfile manager may symlink the file; replace the target, not the link.
    target = edit.path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if edit.private else 0o777)  # existing dirs keep theirs
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600  # new files may hold tokens
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(edit.new.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _tilde(path: Path) -> str:
    """
    Description: Abbreviate paths under the current home directory for display.
    Parameters:
        path: Path to display.
    Output: String using ~ for home or its descendants; home lookup errors may propagate.
    """
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


# --- marked blocks in text files -----------------------------------------------------------------------------------

def _block(marks: _Marks, body: str) -> str:
    """
    Description: Wrap text in ownership markers with consistent trailing newlines.
    Parameters:
        marks: Opening and closing marker pair.
        body: Block content whose trailing newlines are stripped.
    Output: Marked block string ending in a newline.
    """
    return f"{marks.start}\n{body.rstrip(chr(10))}\n{marks.end}\n"


def _regions(text: str, marks: _Marks) -> list[tuple[int, int]]:
    """
    Description: Find complete marked blocks so malformed ownership markers cause a refusal.
    Parameters:
        text: Text to inspect.
        marks: Opening and closing markers to pair.
    Output: List of (start, end) offsets including each end marker newline when present; raises
        RefusalError for lone or unpaired markers.
    """
    spans, pos = [], 0
    while (s := text.find(marks.start, pos)) >= 0:
        e = text.find(marks.end, s + len(marks.start))
        if e < 0:
            raise RefusalError(f"Found '{marks.start}' without its end marker; fix or remove it by hand first.")
        e += len(marks.end)
        e += text[e:e + 1] == "\n"
        spans.append((s, e))
        pos = e
    if text.count(marks.start) != len(spans) or text.count(marks.end) != len(spans):
        raise RefusalError(f"Unpaired '{marks.start}' / '{marks.end}' markers; fix them by hand first.")
    return spans


def _drop(text: str, spans: list[tuple[int, int]]) -> str:
    """
    Description: Remove marked spans and the adjacent blank line inserted by _put.
    Parameters:
        text: Text containing the spans.
        spans: Ordered start/end offsets to remove.
    Output: Text with spans removed; performs no file writes.
    """
    for s, e in reversed(spans):
        before, after = text[:s], text[e:]
        if before == "" and after.startswith("\n"):
            after = after[1:]
        elif before.endswith("\n\n"):
            before = before[:-1]
        text = before + after
    return text


def _put(text: str, block: str, where: str) -> str:
    """
    Description: Place a block at the start or end with separating newlines.
    Parameters:
        text: Existing file text.
        block: Marked block to insert.
        where: start for a leading block, otherwise append.
    Output: Combined text; placing root TOML keys at the start keeps them before tables.
    """
    if where == "start":  # a TOML root key must come before the first [table]
        return block + ("\n" + text if text else "")
    if not text:
        return block
    return text + ("" if text.endswith("\n") else "\n") + "\n" + block


def _drop_all(text: str, *marks: _Marks) -> str:
    """
    Description: Remove the marked blocks of every given marker pair, so an uninstall takes the pre-rename blocks too.
    Parameters:
        text: Existing file text.
        *marks: Marker pairs whose blocks are cut, in order.
    Output: Text without those blocks; raises RefusalError for unpaired markers and writes nothing.
    """
    for pair in marks:
        text = _drop(text, _regions(text, pair))
    return text


def _upsert(text: str, marks: _Marks, blocks: Sequence[tuple[str, str]], legacy: Optional[_Marks] = None) -> str:
    """
    Description: Replace matching owned blocks in place or rebuild their placements when counts
        differ; legacy blocks are rewritten in place only when they are the sole blocks and match
        the new count.
    Parameters:
        text: Existing file text.
        marks: Ownership markers to locate.
        blocks: Sequence of (body, start or end) pairs.
        legacy: Markers of blocks an install from before the rename wrote, or None.
    Output: Updated text; raises RefusalError for unpaired markers and writes nothing.
    """
    if legacy is not None:
        old = _regions(text, legacy)
        if old and not _regions(text, marks) and len(old) == len(blocks):
            for (s, e), (body, _) in reversed(list(zip(old, blocks))):
                text = text[:s] + _block(marks, body) + text[e:]
            return text
        text = _drop(text, old)
    spans = _regions(text, marks)
    if spans and len(spans) == len(blocks):
        for (s, e), (body, _) in reversed(list(zip(spans, blocks))):
            text = text[:s] + _block(marks, body) + text[e:]
        return text
    text = _drop(text, spans)
    for body, where in blocks:
        text = _put(text, _block(marks, body), where)
    return text


def _note_edit(label: str, path: Path, note: str, remove: bool) -> _Edit:
    """
    Description: Plan adding or removing an owned Markdown note.
    Parameters:
        label: Display label for the note file.
        path: Markdown file path.
        note: Note text to insert when installing.
        remove: Whether to remove marked notes instead.
    Output: _Edit describing the change, including deletion if removal empties the file; raises
        RefusalError for unreadable files or malformed markers.
    """
    old = _read(path)
    text = old or ""
    if remove:
        cut = _drop_all(text, MD_MARKS, LEGACY_MD_MARKS)
        new: Optional[str] = old if cut == text else (cut or None)  # a file we emptied goes
    else:
        new = _upsert(text, MD_MARKS, [(note, "end")], LEGACY_MD_MARKS)
    return _Edit(label, path, old, new, remove)


# --- Claude Code: settings.json ------------------------------------------------------------------------------------

def _child(parent: dict, key: str, kind: type, where: str) -> object:
    """
    Description: Get or initialize a typed configuration child while refusing incompatible user
        data.
    Parameters:
        parent: Dictionary whose child is inspected or created.
        key: Child key.
        kind: Expected type and empty-value constructor.
        where: Display location used in refusal messages.
    Output: Existing or newly inserted child; mutates parent for missing keys and raises
        RefusalError for a wrong type.
    """
    value = parent.setdefault(key, kind())
    if not isinstance(value, kind):
        raise RefusalError(f"{where}: '{key}' isn't a {'list' if kind is list else 'JSON object'}; "
                      "fix it by hand first, nothing was written.")
    return value


def _is_ours(hook: object) -> bool:
    """
    Description: Identify hook objects containing the Harpo module command marker.
    Parameters:
        hook: Potential hook object.
    Output: True for a dictionary with a string command containing HOOK_MARK, otherwise False.
    """
    return isinstance(hook, dict) and isinstance(hook.get("command"), str) and HOOK_MARK in hook["command"]


def _has_hooks(entry: object) -> bool:
    """
    Description: Check that an event entry has a hook list before inspecting it.
    Parameters:
        entry: Potential event entry.
    Output: True for a dictionary with a list-valued hooks field, otherwise False.
    """
    return isinstance(entry, dict) and isinstance(entry.get("hooks"), list)


def _strip_event(entries: list, pred: Callable[[object], bool] = _is_ours) -> tuple[list, Optional[int]]:
    """
    Description: Remove selected hooks and drop entries left empty while remembering the first
        removal position.
    Parameters:
        entries: Event entries to filter.
        pred: Predicate selecting hooks to remove, defaulting to Harpo hooks.
    Output: Tuple of retained entries and the first affected insertion position, or None if
        unchanged; predicate errors propagate.
    """
    kept: list = []
    pos: Optional[int] = None
    for entry in entries:
        if _has_hooks(entry) and any(pred(h) for h in entry["hooks"]):
            pos = len(kept) if pos is None else pos
            rest = [h for h in entry["hooks"] if not pred(h)]
            if rest:
                kept.append({**entry, "hooks": rest})
        else:
            kept.append(entry)
    return kept, pos


def _upsert_event(entries: list, want: dict, prior: list[str]) -> tuple[list, list[str], list[str]]:
    """
    Description: Keep exactly one desired Harpo hook entry, replacing any Harpo hook with another
        python path or matcher.
    Parameters:
        entries: Existing event entries.
        want: Desired entry containing one hook.
        prior: Commands owned by an earlier install.
    Output: Tuple of updated entries, owned installed commands, and replaced commands not previously
        owned; malformed desired entries can raise lookup errors.
    """
    cmd = want["hooks"][0]["command"]
    ours = [(i, j) for i, e in enumerate(entries) if _has_hooks(e) for j, h in enumerate(e["hooks"]) if _is_ours(h)]
    if len(ours) == 1:
        i, j = ours[0]
        if entries[i]["hooks"][j] == want["hooks"][0] and entries[i].get("matcher") == want.get("matcher"):
            return entries, [cmd] if cmd in prior else [], []  # not ours unless an earlier install put it there
    replaced = [h["command"] for e in entries if _has_hooks(e) for h in e["hooks"] if _is_ours(h)]
    kept, pos = _strip_event(entries)
    kept.insert(len(kept) if pos is None else pos, want)
    return kept, [cmd], [c for c in replaced if c not in prior]


def _strs(value: object) -> list[str]:
    """
    Description: Filter optional record data to string list entries.
    Parameters:
        value: Potential list of recorded values.
    Output: List containing only strings, or an empty list for nonlists.
    """
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _merge_settings(cur: dict, want: dict, where: str, prior: dict) -> tuple[dict, dict]:
    """
    Description: Merge desired permissions and hooks while recording ownership for conservative
        uninstall; pre-rename MCP allow rules that the record lists are dropped, since the server
        is now `harpo`.
    Parameters:
        cur: Current settings object.
        want: Desired Harpo settings.
        where: Display location for validation errors.
        prior: Earlier installation ownership record.
    Output: Tuple of copied merged settings and ownership record; retains earlier owned entries
        still present and raises RefusalError for incompatible settings structures.
    """
    out, added = copy.deepcopy(cur), {"permissions": {}, "hooks": {}}
    replaced = _strs(prior.get("replaced"))
    perms = _child(out, "permissions", dict, where)
    for kind in ("deny", "allow"):
        rules = _child(perms, kind, list, where)
        before = _strs((prior.get("permissions") or {}).get(kind))
        if kind == "allow":  # the pre-rename tool names an earlier install allowed: they name a server that is gone
            stale = [r for r in rules if r in harness.LEGACY_MCP_TOOLS and r in before]
            rules[:] = [r for r in rules if r not in stale]  # allow rules only: a deny rule is never dropped here
        new = [r for r in want["permissions"][kind] if r not in rules]
        rules.extend(new)
        added["permissions"][kind] = [r for r in before if r in rules and r not in new] + new
    hooks = _child(out, "hooks", dict, where)
    for event, (entry,) in want["hooks"].items():
        before = _strs((prior.get("hooks") or {}).get(event))
        hooks[event], mine, stale = _upsert_event(_child(hooks, event, list, where), entry, before)
        if mine:
            added["hooks"][event] = mine
        replaced += [c for c in stale if c not in replaced]
    return out, {**added, "replaced": replaced}


def _strip_settings(cur: dict, prior: Optional[dict]) -> dict:
    """
    Description: Remove owned settings while preserving identical user-owned rules and hooks.
    Parameters:
        cur: Current settings to copy and filter.
        prior: Earlier ownership record, or None for marker-only hook removal.
    Output: Copied settings without selected entries or newly empty affected tables; without a
        record, only Harpo hook commands go, since permission rules may be the user's own.
    """
    out = copy.deepcopy(cur)
    perms = out.get("permissions")
    if isinstance(perms, dict) and prior is not None:
        touched = False
        for kind in ("deny", "allow"):
            rules, drop = perms.get(kind), _strs((prior.get("permissions") or {}).get(kind))
            if isinstance(rules, list) and any(r in drop for r in rules):
                touched = True
                perms[kind] = [r for r in rules if r not in drop]
                if not perms[kind]:
                    del perms[kind]
        if touched and not perms:
            del out["permissions"]
    hooks = out.get("hooks")
    if isinstance(hooks, dict):
        touched = False
        for event, entries in list(hooks.items()):
            mine = set(_strs((prior.get("hooks") or {}).get(event))) if prior is not None else None
            if isinstance(entries, list):
                kept, pos = _strip_event(
                    entries, lambda h, mine=mine: _is_ours(h) and (mine is None or h["command"] in mine))
                if pos is not None:
                    touched, hooks[event] = True, kept
                    if not kept:
                        del hooks[event]
        if touched and not hooks:
            del out["hooks"]
    return out


def _settings_edit(path: Path, remove: bool, prior: Optional[dict]) -> tuple[_Edit, dict]:
    """
    Description: Plan a validated Claude Code settings update while preserving existing indentation
        and final newline.
    Parameters:
        path: settings.json path.
        remove: Whether to uninstall owned settings.
        prior: Earlier ownership record, or None if absent.
    Output: Tuple of planned _Edit and new ownership record; raises RefusalError for unreadable,
        invalid, or incompatible JSON and writes nothing.
    """
    old, cur = read_json_object(path)
    added: dict = {}
    if remove:
        out = _strip_settings(cur, prior)
    else:
        out, added = _merge_settings(cur, harness.claude_code_settings(), _tilde(path), prior or {})
    if out == cur and old is not None:
        return _Edit("settings.json", path, old, old, remove), added
    if remove and not out:
        return _Edit("settings.json", path, old, None, remove), added  # nothing of the user's left
    return _Edit("settings.json", path, old, dump_json_like(old, out), remove), added


def read_json_object(path: Path) -> tuple[Optional[str], dict]:
    """
    Description: Read a JSON settings file that must hold one object.
    Parameters:
        path: the file.
    Output: (its text, or None when it does not exist; the object, {} when it does not exist). Raises RefusalError for
        a file that cannot be read, is not JSON, or is not an object.
    """
    old = _read(path)
    if old is None:
        return None, {}
    try:
        data = json.loads(old)
    except ValueError as exc:
        raise RefusalError(f"{_tilde(path)} isn't valid JSON ({exc}); fix it first, nothing was written.") from exc
    if not isinstance(data, dict):
        raise RefusalError(f"{_tilde(path)} isn't a JSON object; fix it first, nothing was written.")
    return old, data


def dump_json_like(old: Optional[str], data: dict) -> str:
    """
    Description: Serialize settings with the indentation and final newline of the text they replace.
    Parameters:
        old: the file's current text, or None for a new file.
        data: the object to write.
    Output: The JSON text.
    """
    indent = (re.match(r'\s*\{\s*\n([ \t]+)"', old or "") or [None, "  "])[1]
    tail = "\n" if old is None or old.endswith("\n") else ""
    return json.dumps(data, indent=indent, ensure_ascii=False) + tail


# --- Claude Code: MCP registration ---------------------------------------------------------------------------------

def _mcp_add_argv(claude: str) -> list[str]:
    """
    Description: Build the user-scoped Claude MCP registration invocation for the current
        interpreter.
    Parameters:
        claude: Claude executable path or display name.
    Output: Argument list for claude mcp add; launches nothing.
    """
    command, args = harness.mcp_server()
    return [claude, "mcp", "add", "--scope", "user", MCP_NAME, "--", command, *args]


def _claude(argv: list[str], cwd: Path) -> Optional[subprocess.CompletedProcess]:
    """
    Description: Run a local Claude command with captured output and a bounded wait.
    Parameters:
        argv: Command argument list to run.
        cwd: Working directory for the subprocess.
    Output: CompletedProcess regardless of exit status, or None on OSError or a 60-second timeout;
        launches a subprocess with stdin disconnected.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60, cwd=cwd,
                              stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _is_registered(claude: str, name: str, home: Path) -> Optional[subprocess.CompletedProcess]:
    """
    Description: Ask Claude Code whether an MCP server is registered under a name.
    Parameters:
        claude: Claude executable path.
        name: Server name to look up.
        home: Working directory for the subprocess.
    Output: The `claude mcp get` result when the server is registered, otherwise None (also when `claude` did not
        run); launches a subprocess.
    """
    got = _claude([claude, "mcp", "get", name], home)
    return got if got is not None and got.returncode == 0 else None


def _unregister(claude: str, name: str, home: Path) -> Optional[str]:
    """
    Description: Remove a user-scoped MCP registration through Claude Code.
    Parameters:
        claude: Claude executable path.
        name: Server name to remove.
        home: Working directory for the subprocess.
    Output: None on success, otherwise the first diagnostic line of the failure; launches a subprocess.
    """
    done = _claude([claude, "mcp", "remove", "--scope", "user", name], home)
    return None if done is not None and done.returncode == 0 else _why(done)


def _mcp_step(remove: bool, dry_run: bool, owned: Optional[str]) -> Step:
    """
    Description: Register or remove the Claude MCP server while respecting ownership and dry-run
        mode; an owned pre-rename `harpocrates` is replaced by `harpo` on install, and an unowned
        one is only mentioned.
    Parameters:
        remove: Whether to remove the registration.
        dry_run: Whether to report intended changes without invoking Claude.
        owned: The name the install record says install registered the server under, or None if it did not.
    Output: Step reporting unchanged, manual, added, updated, removed, or failed status; may query
        and change user MCP registration unless dry-run or ownership rules prevent it.
    """
    label = "MCP server"
    claude = shutil.which("claude")
    shown_add = shlex.join(_mcp_add_argv("claude"))
    shown_rm = f"claude mcp remove --scope user {owned or MCP_NAME}"
    shown_old_rm = f"claude mcp remove --scope user {LEGACY_MCP_NAME}"
    if remove and owned is None:  # it may be the user's own registration under our name
        return Step(label, "unchanged", f"not registered by `harpo install`; if it is yours to drop, run: {shown_rm}")
    if claude is None:
        what = (f"if registered, run: {shown_rm}" if remove
                else f"run: {shown_add}" + (f" and then: {shown_old_rm}" if owned == LEGACY_MCP_NAME else ""))
        return Step(label, "manual", f"`claude` isn't on PATH; {what}")
    if dry_run:  # claude isn't invoked at all, not even to look
        return Step(label, "removed" if remove else "added", shown_rm if remove else f"via {shown_add}")
    home = Path.home()
    if remove:
        name = owned or MCP_NAME
        if _is_registered(claude, name, home) is None:
            return Step(label, "unchanged", "not registered")
        why = _unregister(claude, name, home)
        return (Step(label, "removed", "claude mcp remove") if why is None
                else Step(label, "failed", f"{why}; run: {shown_rm}"))
    moved, note = False, ""
    if _is_registered(claude, LEGACY_MCP_NAME, home) is not None:
        if owned == LEGACY_MCP_NAME:  # an install from before the rename made it: the same server under its old name
            why = _unregister(claude, LEGACY_MCP_NAME, home)
            if why is not None:
                return Step(label, "failed", f"{why}; run: {shown_old_rm}")
            moved = True
        else:
            note = (f"; a `{LEGACY_MCP_NAME}` server that `harpo install` did not register is also there, "
                    f"drop it if it is yours: {shown_old_rm}")
    got = _is_registered(claude, MCP_NAME, home)
    registered = got is not None
    command, args = harness.mcp_server()
    # The whole argument list, flag included, so a registration from before the flag is re-registered.
    if got is not None and command in got.stdout and " ".join(args) in got.stdout:
        return (Step(label, "updated", f"dropped the old `{LEGACY_MCP_NAME}` registration" + note) if moved
                else Step(label, "unchanged", "already registered with this python" + note))
    if registered:  # re-register so the python path is current
        _claude([claude, "mcp", "remove", "--scope", "user", MCP_NAME], home)
    done = _claude(_mcp_add_argv(claude), home)
    if done is not None and done.returncode == 0:
        return Step(label, "updated" if registered or moved else "added", "claude mcp add --scope user" + note)
    return Step(label, "failed", f"{_why(done)}; run: {shown_add}")


def _why(done: Optional[subprocess.CompletedProcess]) -> str:
    """
    Description: Summarize a failed Claude invocation for a report.
    Parameters:
        done: Subprocess result, or None when it did not run.
    Output: First diagnostic line truncated to 200 characters, an exit-status fallback, or a
        did-not-run message; whitespace-only diagnostic output can raise IndexError.
    """
    if done is None:
        return "`claude` didn't run"
    return (done.stderr or done.stdout or f"exit {done.returncode}").strip().splitlines()[0][:200]


# --- Codex: config.toml --------------------------------------------------------------------------------------------

def _parse_toml(text: str, path: Path, what: str) -> dict:
    """
    Description: Validate TOML before installation writes any planned configuration.
    Parameters:
        text: TOML text to parse.
        path: Configuration path for diagnostics.
        what: Problem description used on parse failure.
    Output: Parsed dictionary; raises RefusalError when no TOML parser is available or parsing
        fails.
    """
    if tomllib is None:  # pragma: no cover
        raise RefusalError("Python 3.11+ (or the tomli package) is needed to check config.toml before writing it.")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RefusalError(f"{_tilde(path)} {what} ({exc}); nothing was written.") from exc


def _config_edit(path: Path, remove: bool) -> _Edit:
    """
    Description: Plan Codex marked-block changes and refuse conflicts with user-owned settings.
    Parameters:
        path: Codex config.toml path.
        remove: Whether to remove owned blocks instead of installing.
    Output: _Edit with the planned text; raises RefusalError for read failures, malformed markers,
        conflicting settings, invalid TOML, or ineffective generated settings. Writes nothing.
    """
    old = _read(path)
    text = old or ""
    outside = _drop_all(text, TOML_MARKS, LEGACY_TOML_MARKS)
    if remove:
        return _Edit("config.toml", path, old, old if outside == text else (outside or None), remove)
    data = _parse_toml(outside, path, "isn't valid TOML")
    clash = ("default_permissions" if "default_permissions" in data
             else f"[mcp_servers.{MCP_NAME}]" if MCP_NAME in (data.get("mcp_servers") or {})
             else f"a [permissions.{MCP_NAME}] table" if MCP_NAME in (data.get("permissions") or {}) else None)
    if clash:
        raise RefusalError(f"{_tilde(path)} already defines {clash} outside the Harpo block. Merge by hand "
                      "(`harpo setup codex` prints the lines) or remove yours, then re-run; nothing was written.")
    root, tables = harness.codex_config_parts()
    top = "# default_permissions must stay above the first [table]; the rest of this block is at the end.\n" + root
    new = _upsert(text, TOML_MARKS, [(top, "start"), (tables, "end")], LEGACY_TOML_MARKS)
    merged = _parse_toml(new, path, "would not be valid TOML with Harpo's lines")
    if merged.get("default_permissions") != MCP_NAME or MCP_NAME not in merged.get("mcp_servers", {}):
        raise RefusalError(f"{_tilde(path)} would not take Harpo's lines as written; nothing was written.")
    return _Edit("config.toml", path, old, new, remove)


# --- orchestration -------------------------------------------------------------------------------------------------

def _plan(target: str, remove: bool, home: Path, prior: Optional[dict]) -> tuple[list[_Edit], dict]:
    """
    Description: Plan all harness file edits before any write to avoid writing after a configuration
        refusal.
    Parameters:
        target: Harness identifier selecting Claude Code or Codex configuration.
        remove: Whether to plan removal.
        home: Home directory containing harness files.
        prior: Earlier ownership record, or None.
    Output: Tuple of edits and target ownership record; marked blocks require no ownership record.
        RefusalError propagates from unsafe plans.
    """
    if target == "claude-code":
        base = home / ".claude"
        settings, added = _settings_edit(base / "settings.json", remove, prior)
        note = _note_edit("CLAUDE.md", base / "CLAUDE.md", harness.CLAUDE_NOTE, remove)
        return [settings, note], {**added, "blocks": ["CLAUDE.md"]}
    base = home / ".codex"
    return ([_config_edit(base / "config.toml", remove),
             _note_edit("AGENTS.md", base / "AGENTS.md", harness.CODEX_NOTE, remove)],
            {"blocks": ["config.toml", "AGENTS.md"]})


def _load_record(path: Path) -> tuple[Optional[str], dict, Optional[str]]:
    """
    Description: Read the installation record without treating an unreadable record as an empty
        valid one.
    Parameters:
        path: Installation record path.
    Output: Tuple of file text or None, record dictionary, and problem string or None; missing files
        have no problem, while read and JSON failures are reported.
    """
    try:
        text = _read(path)
    except RefusalError as exc:
        return None, {}, str(exc)
    if text is None:
        return None, {}, None
    try:
        data = json.loads(text)
    except ValueError as exc:
        return text, {}, f"{_tilde(path)} isn't valid JSON ({exc})."
    if not isinstance(data, dict):
        return text, {}, f"{_tilde(path)} isn't a JSON object."
    return text, data, None


def detected_targets() -> list[str]:
    """
    Description: Select supported install targets that appear installed.
    Parameters: None
    Output: List of detected target identifiers in TARGETS order; detection errors may propagate.
    """
    found = harness.detect()
    return [t for t in TARGETS if found.get(NAMES[t])]


def _apply(edit: _Edit, dry_run: bool, report: Report) -> None:
    """
    Description: Apply or preview one edit and append its outcome to a report.
    Parameters:
        edit: Planned file edit.
        dry_run: Whether to skip actual writes.
        report: Report to receive the resulting step.
    Output: None; appends a step and may write the edit. Records OSError failures as failed steps;
        other exceptions propagate.
    """
    try:
        if edit.changed and not dry_run:
            _write(edit)
        detail = _tilde(edit.path) + (f" (original kept as {edit.backup.name})" if edit.needs_backup else "")
        report.steps.append(Step(edit.label, edit.state, detail))
    except OSError as exc:
        report.steps.append(Step(edit.label, "failed", f"{_tilde(edit.path)}: {exc.strerror or exc}"))


def _run(targets: Sequence[str], dry_run: bool, remove: bool) -> list[Report]:
    """
    Description: Coordinate planned harness edits and record ownership last so removal can preserve
        user-owned configuration.
    Parameters:
        targets: Harness identifiers to process in order.
        dry_run: Whether to preview edits without applying them or invoking Claude.
        remove: Whether to remove installed configuration.
    Output: List of reports containing outcomes and refusals; may write harness files, backups, MCP
        registration, and private install state. The state lookup may copy the old state folder to the new
        one, except in a dry run, which has no side effects (SEC-09); unhandled filesystem or data-shape
        errors propagate.
    """
    home = Path.home()
    state = paths.peek_state_dir() if dry_run else state_dir()  # a dry run looks at the state folder, it never moves it
    reports = [Report(t) for t in targets]
    try:
        paths.require_usable(state)  # a symlink or a file where ~/.harpo should be: no record read, none written
    except paths.StateDirUnusableError as exc:
        for report in reports:
            report.refused = str(exc)
        return reports
    rpath = state / "install.json"
    rtext, record, problem = _load_record(rpath)
    if problem and not remove:  # without it a later uninstall couldn't tell ours from theirs
        for report in reports:
            report.refused = f"Install record unreadable: {problem} Fix or delete {_tilde(rpath)}, then re-run."
        return reports
    new_record = dict(record)
    for target, report in zip(targets, reports):
        prior = record.get(target) if isinstance(record.get(target), dict) else None
        try:
            edits, section = _plan(target, remove, home, prior)  # all planned before any write: no half install
        except RefusalError as exc:
            report.refused = str(exc)
            continue
        before = len(report.steps)
        for edit in edits:
            _apply(edit, dry_run, report)
        if any(s.state == "failed" for s in report.steps[before:]):
            continue  # the record stays as it was
        # The name install registered the server under: "mcp_name" in the record, `harpocrates` for a record from
        # before the rename (it has "mcp": true and no name), None when the registration is not install's.
        owned = (None if not (prior and prior.get("mcp")) else
                 prior["mcp_name"] if isinstance(prior.get("mcp_name"), str) else LEGACY_MCP_NAME)
        if target == "claude-code":
            step = _mcp_step(remove, dry_run, owned)
            report.steps.insert(1, step)
            # Install owns the registration if it made one, or already did and nothing took it away.
            kept = ((owned if step.state in ("failed", "manual") else None) if remove
                    else MCP_NAME if step.state in ("added", "updated") else owned)
            section["mcp"] = kept is not None
            if kept is not None:
                section["mcp_name"] = kept
            if remove and prior is None and not problem:
                report.notes.append(NO_RECORD_NOTE)
        elif not remove:
            report.notes.append(HOOKS_TRUST_NOTE)
        if not remove:
            new_record[target] = section
        elif section.get("mcp"):
            new_record[target] = {"mcp": True, "mcp_name": section["mcp_name"]}  # removal failed or `claude` was missing: keep the claim for a retry
        else:
            new_record.pop(target, None)
    if problem:
        for report in reports:
            report.notes.append(f"Install record unreadable ({problem}), so only marked blocks and Harpo hook "
                                "commands were removed; it was left in place.")
        return reports
    sections = {t: new_record[t] for t in TARGETS if isinstance(new_record.get(t), dict)}
    # "python" is the interpreter that runs the hooks. The Claude Code prompt plugin runs the CLI through it
    # (never a bare `harpo` from PATH). Records from before it existed stay valid without it.
    # "python_args" are the flags it runs with (harness.python_flags: drop the agent-controlled cwd from sys.path and
    # ignore PYTHON* variables), one list item per flag.
    python = record.get("python") if remove else sys.executable
    python_args = record.get("python_args") if remove else harness.python_flags()
    extra: dict = {}
    if isinstance(python, str) and python:
        extra["python"] = python
        if isinstance(python_args, list):
            extra["python_args"] = python_args
    if "gate" in record:  # written by `harpo guard up`; dropping it would switch the hook's SEC-10 check off
        extra["gate"] = record["gate"]
    text = json.dumps({"version": 1, **extra, **sections}, indent=2) + "\n" if sections or "gate" in record else None
    host = next((r for r in reversed(reports) if not r.refused), None)
    if host is not None:
        _apply(_Edit("install record", rpath, rtext, text, remove, private=True), dry_run, host)
    return reports


def install_harnesses(targets: Sequence[str], dry_run: bool = False) -> list[Report]:
    """
    Description: Install read-tool configuration for selected harnesses and track what was added.
    Parameters:
        targets: Harness setup identifiers to install.
        dry_run: Whether to preview rather than apply changes.
    Output: List of reports; may write config, backups, and ownership state or run Claude MCP
        registration unless dry-run. Unhandled orchestration errors propagate.
    """
    return _run(targets, dry_run, remove=False)


def uninstall_harnesses(targets: Sequence[str], dry_run: bool = False) -> list[Report]:
    """
    Description: Remove owned read-tool configuration while retaining user-owned settings and
        backups.
    Parameters:
        targets: Harness setup identifiers to uninstall.
        dry_run: Whether to preview rather than apply changes.
    Output: List of reports; may change config, ownership state, and owned MCP registration unless
        dry-run. Unhandled orchestration errors propagate.
    """
    return _run(targets, dry_run, remove=True)


def failed(reports: Sequence[Report]) -> bool:
    """
    Description: Check reports for any refused operation or failed step.
    Parameters:
        reports: Operation reports to inspect.
    Output: True if any report is refused or contains a failed step, otherwise False.
    """
    return any(r.refused or any(s.state == "failed" for s in r.steps) for r in reports)


_WOULD = {"added": "would add", "updated": "would update", "removed": "would remove"}


def render(reports: Sequence[Report], dry_run: bool, remove: bool) -> str:
    """
    Description: Format operation reports with dry-run wording and removal notes.
    Parameters:
        reports: Reports to display.
        dry_run: Whether to describe proposed changes.
        remove: Whether to include the retained-backups note.
    Output: Human-readable report string; raises KeyError for unknown report targets and performs no
        writes.
    """
    lines = ["Dry run: nothing is written and `claude` is not run."] if dry_run else []
    for report in reports:
        lines.append(f"{NAMES[report.target]}")
        for step in report.steps:
            state = _WOULD.get(step.state, step.state) if dry_run else step.state
            lines.append(f"  {step.label:<15}{state:<14}{step.detail}".rstrip())
        if report.refused:
            lines.append(f"  refused: {report.refused}")
        lines += [f"  {note}" for note in report.notes]
    if remove:
        lines.append("Backups (*" + BACKUP_SUFFIX + ", and *" + LEGACY_BACKUP_SUFFIX
                     + " from an install before the rename) are left in place.")
    return "\n".join(lines)

