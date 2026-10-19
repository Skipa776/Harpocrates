"""FR-INSTALL-01: write the read-tool setup into each harness's own files, and take it out again.

Claude Code: ~/.claude/settings.json (deny/allow rules and hooks, merged), the harpocrates MCP server (through
`claude mcp`), and a marked note in ~/.claude/CLAUDE.md. Codex: a marked block in ~/.codex/config.toml and a
marked note in ~/.codex/AGENTS.md. Every file is planned in memory first, so a refusal (unparsable file, a clash
with the user's own config) writes nothing for that harness. Writes are atomic, keep the file mode, and the first
original is kept once as <file>.harpocrates.bak (FR-INSTALL-08). Nothing here talks to the network.
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

from Harpocrates import harness
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
BACKUP_SUFFIX = ".harpocrates.bak"
HOOK_MARK = "-m Harpocrates.hooks"  # in the command of every hook entry we write, whatever the python path
MCP_NAME = "harpocrates"
HOOKS_TRUST_NOTE = (
    "One manual step: Codex runs a new or changed hook only after you trust it. Start `codex`, type /hooks, "
    "review the Harpocrates PreToolUse hook and trust it (once)."
)


NO_RECORD_NOTE = (
    "No install record (~/.harpo/install.json): removed only Harpocrates' marked blocks and hook commands. "
    "Permission rules and the MCP server are left, since they may be yours; drop them by hand if not."
)


class RefusalError(Exception):
    """The install can't proceed safely; the message says why and what to do. Nothing was written."""


@dataclass(frozen=True)
class Step:
    label: str
    state: str  # added | updated | removed | unchanged | manual | failed
    detail: str = ""


@dataclass
class Report:
    target: str
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    refused: Optional[str] = None


@dataclass(frozen=True)
class _Marks:
    start: str
    end: str


MD_MARKS = _Marks("<!-- harpocrates:start -->", "<!-- harpocrates:end -->")
TOML_MARKS = _Marks("# >>> harpocrates >>>", "# <<< harpocrates <<<")


@dataclass(frozen=True)
class _Edit:
    label: str
    path: Path
    old: Optional[str]  # None: the file doesn't exist
    new: Optional[str]  # None: delete the file
    remove: bool
    private: bool = False  # the install record: dir 0700, no backup copy

    @property
    def changed(self) -> bool:
        return self.new != self.old

    @property
    def state(self) -> str:
        if not self.changed:
            return "unchanged"
        return "removed" if self.remove else ("added" if self.old is None else "updated")

    @property
    def backup(self) -> Path:
        return self.path.with_name(self.path.name + BACKUP_SUFFIX)

    @property
    def needs_backup(self) -> bool:
        return self.changed and self.old is not None and not self.private and not self.backup.exists()


# --- reading and writing files -------------------------------------------------------------------------------------

def _read(path: Path) -> Optional[str]:
    try:
        return path.read_bytes().decode("utf-8")  # bytes: keeps CRLF and a missing final newline as they are
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusalError(f"Can't read {_tilde(path)} ({exc}); nothing was written.") from exc


def _write(edit: _Edit) -> None:
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
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


# --- marked blocks in text files -----------------------------------------------------------------------------------

def _block(marks: _Marks, body: str) -> str:
    return f"{marks.start}\n{body.rstrip(chr(10))}\n{marks.end}\n"


def _regions(text: str, marks: _Marks) -> list[tuple[int, int]]:
    """Spans of the marked blocks, each through its end marker's newline. A lone marker is a refusal."""
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
    """Cut the blocks and the one blank line _put added next to each."""
    for s, e in reversed(spans):
        before, after = text[:s], text[e:]
        if before == "" and after.startswith("\n"):
            after = after[1:]
        elif before.endswith("\n\n"):
            before = before[:-1]
        text = before + after
    return text


def _put(text: str, block: str, where: str) -> str:
    if where == "start":  # a TOML root key must come before the first [table]
        return block + ("\n" + text if text else "")
    if not text:
        return block
    return text + ("" if text.endswith("\n") else "\n") + "\n" + block


def _upsert(text: str, marks: _Marks, blocks: Sequence[tuple[str, str]]) -> str:
    """Replace our blocks in place, or add them: (body, "start" | "end") each."""
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
    old = _read(path)
    text = old or ""
    if remove:
        cut = _drop(text, _regions(text, MD_MARKS))
        new: Optional[str] = old if cut == text else (cut or None)  # a file we emptied goes
    else:
        new = _upsert(text, MD_MARKS, [(note, "end")])
    return _Edit(label, path, old, new, remove)


# --- Claude Code: settings.json ------------------------------------------------------------------------------------

def _child(parent: dict, key: str, kind: type, where: str) -> object:
    value = parent.setdefault(key, kind())
    if not isinstance(value, kind):
        raise RefusalError(f"{where}: '{key}' isn't a {'list' if kind is list else 'JSON object'}; "
                      "fix it by hand first, nothing was written.")
    return value


def _is_ours(hook: object) -> bool:
    return isinstance(hook, dict) and isinstance(hook.get("command"), str) and HOOK_MARK in hook["command"]


def _has_hooks(entry: object) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("hooks"), list)


def _strip_event(entries: list, pred: Callable[[object], bool] = _is_ours) -> tuple[list, Optional[int]]:
    """The event's entries without the hooks `pred` picks (an entry left empty goes), and where the first sat."""
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
    """(entries, hook commands install added, Harpocrates commands it replaced). Our entry stays once: a
    Harpocrates hook with another python path (or matcher) is replaced. `prior` is what an earlier install added."""
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
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _merge_settings(cur: dict, want: dict, where: str, prior: dict) -> tuple[dict, dict]:
    """(merged settings, what this install added). The record is what lets uninstall leave the user's own
    identical rules and hooks alone; entries an earlier install added stay in it while they are still there."""
    out, added = copy.deepcopy(cur), {"permissions": {}, "hooks": {}}
    replaced = _strs(prior.get("replaced"))
    perms = _child(out, "permissions", dict, where)
    for kind in ("deny", "allow"):
        rules = _child(perms, kind, list, where)
        new = [r for r in want["permissions"][kind] if r not in rules]
        before = _strs((prior.get("permissions") or {}).get(kind))
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
    """Undo _merge_settings. With the install record (`prior`) only what it lists goes; without one only
    Harpocrates hook commands go, never permission rules (they may be the user's own). A list or table that
    ends up empty goes with them."""
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
    """(the edit, this install's record for the file). `prior` is the earlier record, None if there is none."""
    old = _read(path)
    cur: dict = {}
    if old is not None:
        try:
            cur = json.loads(old)
        except ValueError as exc:
            raise RefusalError(f"{_tilde(path)} isn't valid JSON ({exc}); fix it first, nothing was written.") from exc
        if not isinstance(cur, dict):
            raise RefusalError(f"{_tilde(path)} isn't a JSON object; fix it first, nothing was written.")
    added: dict = {}
    if remove:
        out = _strip_settings(cur, prior)
    else:
        out, added = _merge_settings(cur, harness.claude_code_settings(), _tilde(path), prior or {})
    if out == cur and old is not None:
        return _Edit("settings.json", path, old, old, remove), added
    if remove and not out:
        return _Edit("settings.json", path, old, None, remove), added  # nothing of the user's left
    indent = (re.match(r'\s*\{\s*\n([ \t]+)"', old or "") or [None, "  "])[1]  # keep their indentation
    tail = "\n" if old is None or old.endswith("\n") else ""
    text = json.dumps(out, indent=indent, ensure_ascii=False) + tail
    return _Edit("settings.json", path, old, text, remove), added


# --- Claude Code: MCP registration ---------------------------------------------------------------------------------

def _mcp_add_argv(claude: str) -> list[str]:
    command, args = harness.mcp_server()
    return [claude, "mcp", "add", "--scope", "user", MCP_NAME, "--", command, *args]


def _claude(argv: list[str], cwd: Path) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60, cwd=cwd,
                              stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _mcp_step(remove: bool, dry_run: bool, owned: bool) -> Step:
    """`owned`: the install record says install registered the server (only used on removal)."""
    label = "MCP server"
    claude = shutil.which("claude")
    shown_add = shlex.join(_mcp_add_argv("claude"))
    shown_rm = f"claude mcp remove --scope user {MCP_NAME}"
    if remove and not owned:  # it may be the user's own registration under our name
        return Step(label, "unchanged", f"not registered by `harpocrates install`; if it is yours to drop, run: {shown_rm}")
    if claude is None:
        what = f"if registered, run: {shown_rm}" if remove else f"run: {shown_add}"
        return Step(label, "manual", f"`claude` isn't on PATH; {what}")
    if dry_run:  # claude isn't invoked at all, not even to look
        return Step(label, "removed" if remove else "added", shown_rm if remove else f"via {shown_add}")
    home = Path.home()
    got = _claude([claude, "mcp", "get", MCP_NAME], home)
    registered = got is not None and got.returncode == 0
    if remove:
        if not registered:
            return Step(label, "unchanged", "not registered")
        done = _claude([claude, "mcp", "remove", "--scope", "user", MCP_NAME], home)
        return (Step(label, "removed", "claude mcp remove") if done and done.returncode == 0
                else Step(label, "failed", f"{_why(done)}; run: {shown_rm}"))
    command, args = harness.mcp_server()
    # The whole argument list, flag included, so a registration from before the flag is re-registered.
    if registered and got is not None and command in got.stdout and " ".join(args) in got.stdout:
        return Step(label, "unchanged", "already registered with this python")
    if registered:  # re-register so the python path is current
        _claude([claude, "mcp", "remove", "--scope", "user", MCP_NAME], home)
    done = _claude(_mcp_add_argv(claude), home)
    if done is not None and done.returncode == 0:
        return Step(label, "updated" if registered else "added", "claude mcp add --scope user")
    return Step(label, "failed", f"{_why(done)}; run: {shown_add}")


def _why(done: Optional[subprocess.CompletedProcess]) -> str:
    if done is None:
        return "`claude` didn't run"
    return (done.stderr or done.stdout or f"exit {done.returncode}").strip().splitlines()[0][:200]


# --- Codex: config.toml --------------------------------------------------------------------------------------------

def _parse_toml(text: str, path: Path, what: str) -> dict:
    if tomllib is None:  # pragma: no cover
        raise RefusalError("Python 3.11+ (or the tomli package) is needed to check config.toml before writing it.")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RefusalError(f"{_tilde(path)} {what} ({exc}); nothing was written.") from exc


def _config_edit(path: Path, remove: bool) -> _Edit:
    old = _read(path)
    text = old or ""
    spans = _regions(text, TOML_MARKS)
    outside = _drop(text, spans)
    if remove:
        return _Edit("config.toml", path, old, old if outside == text else (outside or None), remove)
    data = _parse_toml(outside, path, "isn't valid TOML")
    clash = ("default_permissions" if "default_permissions" in data
             else "[mcp_servers.harpocrates]" if MCP_NAME in (data.get("mcp_servers") or {})
             else "a [permissions.harpocrates] table" if MCP_NAME in (data.get("permissions") or {}) else None)
    if clash:
        raise RefusalError(f"{_tilde(path)} already defines {clash} outside the Harpocrates block. Merge by hand "
                      "(`harpocrates setup codex` prints the lines) or remove yours, then re-run; nothing was written.")
    root, tables = harness.codex_config_parts()
    top = "# default_permissions must stay above the first [table]; the rest of this block is at the end.\n" + root
    new = _upsert(text, TOML_MARKS, [(top, "start"), (tables, "end")])
    merged = _parse_toml(new, path, "would not be valid TOML with Harpocrates' lines")
    if merged.get("default_permissions") != "harpocrates" or MCP_NAME not in merged.get("mcp_servers", {}):
        raise RefusalError(f"{_tilde(path)} would not take Harpocrates' lines as written; nothing was written.")
    return _Edit("config.toml", path, old, new, remove)


# --- orchestration -------------------------------------------------------------------------------------------------

def _plan(target: str, remove: bool, home: Path, prior: Optional[dict]) -> tuple[list[_Edit], dict]:
    """(edits, this install's record for the target). Marked blocks need no record: the markers are theirs."""
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
    """(file text, record, problem). A record that can't be read is a problem, not an empty record."""
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
    found = harness.detect()
    return [t for t in TARGETS if found.get(NAMES[t])]


def _apply(edit: _Edit, dry_run: bool, report: Report) -> None:
    try:
        if edit.changed and not dry_run:
            _write(edit)
        detail = _tilde(edit.path) + (f" (original kept as {edit.backup.name})" if edit.needs_backup else "")
        report.steps.append(Step(edit.label, edit.state, detail))
    except OSError as exc:
        report.steps.append(Step(edit.label, "failed", f"{_tilde(edit.path)}: {exc.strerror or exc}"))


def _run(targets: Sequence[str], dry_run: bool, remove: bool) -> list[Report]:
    """The install record (RECORD) says what install added, so uninstall removes only that and leaves the
    user's own identical rules, hooks and MCP registration. It is written last, after the files it describes."""
    home = Path.home()
    rpath = state_dir() / "install.json"
    rtext, record, problem = _load_record(rpath)
    reports = [Report(t) for t in targets]
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
        owned = bool(prior and prior.get("mcp"))
        if target == "claude-code":
            step = _mcp_step(remove, dry_run, owned)
            report.steps.insert(1, step)
            # Install owns the registration if it made one, or already did and nothing took it away.
            section["mcp"] = (owned and step.state in ("failed", "manual") if remove
                              else step.state in ("added", "updated") or owned)
            if remove and prior is None and not problem:
                report.notes.append(NO_RECORD_NOTE)
        elif not remove:
            report.notes.append(HOOKS_TRUST_NOTE)
        if not remove:
            new_record[target] = section
        elif section.get("mcp"):
            new_record[target] = {"mcp": True}  # removal failed or `claude` was missing: keep the claim for a retry
        else:
            new_record.pop(target, None)
    if problem:
        for report in reports:
            report.notes.append(f"Install record unreadable ({problem}), so only marked blocks and Harpocrates hook "
                                "commands were removed; it was left in place.")
        return reports
    sections = {t: new_record[t] for t in TARGETS if isinstance(new_record.get(t), dict)}
    # "python" is the interpreter that runs the hooks. The Claude Code prompt plugin runs the CLI through it
    # (never a bare `harpocrates` from PATH). Records from before it existed stay valid without it.
    # "python_args" are the flags it runs with (harness.python_flag: drop the agent-controlled cwd from sys.path).
    python = record.get("python") if remove else sys.executable
    python_args = record.get("python_args") if remove else [harness.python_flag()]
    extra: dict = {}
    if isinstance(python, str) and python:
        extra["python"] = python
        if isinstance(python_args, list):
            extra["python_args"] = python_args
    text = json.dumps({"version": 1, **extra, **sections}, indent=2) + "\n" if sections else None
    host = next((r for r in reversed(reports) if not r.refused), None)
    if host is not None:
        _apply(_Edit("install record", rpath, rtext, text, remove, private=True), dry_run, host)
    return reports


def install_harnesses(targets: Sequence[str], dry_run: bool = False) -> list[Report]:
    return _run(targets, dry_run, remove=False)


def uninstall_harnesses(targets: Sequence[str], dry_run: bool = False) -> list[Report]:
    return _run(targets, dry_run, remove=True)


def failed(reports: Sequence[Report]) -> bool:
    return any(r.refused or any(s.state == "failed" for s in r.steps) for r in reports)


_WOULD = {"added": "would add", "updated": "would update", "removed": "would remove"}


def render(reports: Sequence[Report], dry_run: bool, remove: bool) -> str:
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
        lines.append("Backups (*" + BACKUP_SUFFIX + ") are left in place.")
    return "\n".join(lines)

