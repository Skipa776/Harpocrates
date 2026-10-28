"""G7: route Claude Code and Codex through the egress gate, take them off it, and check the result on this machine.

`harpo guard up` starts the gate in the background with a new token (gate decision 11) and, once it answers, points each
harness at it; the install record's gate entry is written last, which turns on the hook's SEC-10 check. `harpo guard down`
undoes that in reverse. A file guard up wrote and nobody changed since gets its exact earlier bytes back from a snapshot
(FR-INSTALL-08); a file changed since loses only Harpo's lines. A harness that already routes elsewhere is refused:
chaining to an existing gateway is FR-INSTALL-07 (M3). `harpo doctor` checks all of this locally and sends nothing to a
provider. The design is "G7" in docs/design.md (FR-INSTALL-01, -02, -06, -08, FR-UX-01, SEC-03).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from Harpocrates import audit, gatecheck, harness, install, paths
from Harpocrates.gate import session_token
from Harpocrates.install import RefusalError

TOKEN_HEADER = session_token.TOKEN_HEADER
DEFAULT_PORT = 47321
PROVIDER = "harpo"  # the Codex model provider guard up adds
CLAUDE_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS")
GATE_MARKS = install._Marks("# >>> harpo gate >>>", "# <<< harpo gate <<<")
START_TIMEOUT = 10.0
STOP_TIMEOUT = 5.0
_OUR_URL = re.compile(r"http://127\.0\.0\.1:[0-9]{1,5}")


@dataclass(frozen=True)
class Files:
    """
    Description: The gate's files in the state folder (~/.harpo), all private to the user.
    Parameters:
        state: the state folder.
    Output: Paths for the token, pid, request log, listening line, snapshots and install record.
    """
    state: Path

    @property
    def token(self) -> Path:
        """
        Description: The gate token (0600), read by the gate at start.
        Parameters: None
        Output: state/gate.token.
        """
        return self.state / "gate.token"

    @property
    def pid(self) -> Path:
        """
        Description: The running gate's pid and port.
        Parameters: None
        Output: state/gate.pid.
        """
        return self.state / "gate.pid"

    @property
    def log(self) -> Path:
        """
        Description: The gate's request log: one line per request, fixed fields, never a header value or body.
        Parameters: None
        Output: state/gate.log.
        """
        return self.state / "gate.log"

    @property
    def out(self) -> Path:
        """
        Description: Where the gate prints its listening line.
        Parameters: None
        Output: state/gate.out.
        """
        return self.state / "gate.out"

    @property
    def snapshots(self) -> Path:
        """
        Description: The text each routed file had before guard up, to restore on guard down.
        Parameters: None
        Output: state/gate-snapshots.
        """
        return self.state / "gate-snapshots"

    @property
    def record(self) -> Path:
        """
        Description: The install record shared with `harpo install`.
        Parameters: None
        Output: state/install.json.
        """
        return self.state / "install.json"


@dataclass
class Outcome:
    """
    Description: What a guard command did, for the CLI to print.
    Parameters:
        lines: report lines, in order.
        failed: whether the command failed (exit status 1).
    Output: A mutable record.
    """
    lines: list[str] = field(default_factory=list)
    failed: bool = False

    def fail(self, line: str) -> "Outcome":
        """
        Description: Add a failure line and mark the outcome failed.
        Parameters:
            line: what went wrong and what to do.
        Output: self.
        """
        self.lines.append(line)
        self.failed = True
        return self


@dataclass(frozen=True)
class Check:
    """
    Description: One `harpo doctor` result.
    Parameters:
        name: what was checked.
        status: ok, warn or fail.
        detail: what was found and, for a failure, how to fix it.
    Output: An immutable record.
    """
    name: str
    status: str
    detail: str


# --- the routing each harness gets ---------------------------------------------------------------------------------

def _sha(text: Optional[str]) -> Optional[str]:
    """
    Description: Fingerprint a file's text, so guard down can tell whether anyone changed what guard up wrote.
    Parameters:
        text: the text, or None for a missing file.
    Output: Its SHA-256 in hex, or None.
    """
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ours(name: str, value: object) -> bool:
    """
    Description: Whether a Claude Code env value is one guard up writes.
    Parameters:
        name: ANTHROPIC_BASE_URL or ANTHROPIC_CUSTOM_HEADERS.
        value: its value.
    Output: True for the loopback gate URL, or for one header line carrying the gate token.
    """
    if not isinstance(value, str):
        return False
    if name == "ANTHROPIC_BASE_URL":
        return bool(_OUR_URL.fullmatch(value))
    return value.startswith(TOKEN_HEADER + ": ") and "\n" not in value


def plan_claude(path: Path, port: Optional[int], token: str = "") -> tuple[Optional[str], Optional[str]]:
    """
    Description: Plan Claude Code's settings.json with the gate routing added, or with Harpo's routing taken out.
    Parameters:
        path: ~/.claude/settings.json.
        port: the gate's port, or None to plan the removal.
        token: the gate token (adding only).
    Output: (current text or None, new text). A removal with nothing of Harpo's in the file returns the text unchanged.
        Raises RefusalError, writing nothing, for a file that is not a JSON object, an env that is not an object, or, when
        adding, an ANTHROPIC_BASE_URL or ANTHROPIC_CUSTOM_HEADERS that Harpo did not write or a CLAUDE_CODE_USE_<provider>
        that is on.
    """
    old, data = install.read_json_object(path)
    env = data.get("env", {})
    if not isinstance(env, dict):
        raise RefusalError(f"{install._tilde(path)} has an env that is not an object; fix it first, nothing was written.")
    if port is not None:
        for name in CLAUDE_VARS:
            if name in env and not _ours(name, env[name]):
                raise RefusalError(f"{install._tilde(path)} already sets {name}, so Claude Code is routed elsewhere. "
                                   "The gate cannot chain to another gateway yet (FR-INSTALL-07).")
        flags = [name for name, value in env.items() if gatecheck.provider_flag_on(name, value)]
        if flags:
            raise RefusalError(f"{install._tilde(path)} sets {flags[0]}: Claude Code uses another model provider, "
                               "which the gate cannot route.")
    kept = {k: v for k, v in env.items() if not (k in CLAUDE_VARS and _ours(k, v))}
    if port is None and kept == env:
        return old, old
    if port is not None:
        kept.update({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                     "ANTHROPIC_CUSTOM_HEADERS": f"{TOKEN_HEADER}: {token}"})
    out = {k: v for k, v in data.items() if k != "env"}
    if kept:
        out = {k: (kept if k == "env" else v) for k, v in data.items()} if "env" in data else {**out, "env": kept}
    return old, install.dump_json_like(old, out)


def plan_codex(path: Path, port: Optional[int], token: str = "") -> tuple[Optional[str], Optional[str]]:
    """
    Description: Plan Codex's config.toml with Harpo's gate block added (route A: a custom provider), or taken out.
    Parameters:
        path: ~/.codex/config.toml.
        port: the gate's port, or None to plan the removal.
        token: the gate token (adding only).
    Output: (current text or None, new text). Raises RefusalError, writing nothing, for invalid TOML, unpaired markers,
        or, when adding, a model_provider, openai_base_url or [model_providers.harpo] outside Harpo's block.
    """
    old = install._read(path)
    text = old or ""
    outside = install._drop_all(text, GATE_MARKS)
    if port is None:
        return old, (old if outside == text else outside)
    data = install._parse_toml(outside, path, "isn't valid TOML")
    clash = next((key for key in ("model_provider", "openai_base_url") if key in data), None)
    if clash is None and PROVIDER in (data.get("model_providers") or {}):
        clash = f"[model_providers.{PROVIDER}]"
    if clash:
        raise RefusalError(f"{install._tilde(path)} already sets {clash}, so Codex is routed elsewhere. "
                           "The gate cannot chain to another gateway yet (FR-INSTALL-07).")
    url = f"http://127.0.0.1:{port}/v1"
    root = f'model_provider = "{PROVIDER}"  # Codex sends every request to the Harpo gate\n'
    table = (f'[model_providers.{PROVIDER}]\nname = "Harpo gate"\nbase_url = "{url}"\nwire_api = "responses"\n'
             f'requires_openai_auth = true\nhttp_headers = {{ "{TOKEN_HEADER}" = "{token}" }}\n')
    new = install._upsert(text, GATE_MARKS, [(root, "start"), (table, "end")])
    merged = install._parse_toml(new, path, "would not be valid TOML with Harpo's gate lines")
    if merged.get("model_provider") != PROVIDER or merged["model_providers"][PROVIDER].get("base_url") != url:
        raise RefusalError(f"{install._tilde(path)} would not take Harpo's gate lines as written; nothing was written.")
    return old, new


PLANS: dict[str, tuple[str, Callable[[Path, Optional[int], str], tuple[Optional[str], Optional[str]]]]] = {
    "claude-code": (".claude/settings.json", plan_claude),
    "codex": (".codex/config.toml", plan_codex),
}


# --- the gate process ----------------------------------------------------------------------------------------------

def _private(path: Path, text: str) -> None:
    """
    Description: Write a private state file (mode 0600, folder 0700) atomically.
    Parameters:
        path: the file.
        text: its contents.
    Output: None.
    """
    install._write(install._Edit(path.name, path, None, text, False, private=True))


def _cmdline(pid: int) -> str:
    """
    Description: The command line of a process, to make sure a pid is still the gate before signalling it.
    Parameters:
        pid: the process id.
    Output: The command line, or "" when the process is gone or cannot be inspected.
    """
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        pass
    try:  # macOS has no /proc
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                              timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def running_gate(files: Files) -> Optional[tuple[int, int]]:
    """
    Description: Find the gate that guard up started.
    Parameters:
        files: the state files.
    Output: (pid, port) when gate.pid names a live Harpo gate, otherwise None.
    """
    try:
        info = json.loads(files.pid.read_text(encoding="utf-8"))
        pid, port = int(info["pid"]), int(info["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (pid, port) if "Harpocrates.gate" in _cmdline(pid) else None


def _reaped(pid: int) -> bool:
    """
    Description: Whether a process has exited (reaping it if it is a child of this one).
    Parameters:
        pid: the process id.
    Output: True once it is gone.
    """
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return True
    except ChildProcessError:  # not our child: a gate from an earlier guard up
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def stop_gate(files: Files) -> Optional[int]:
    """
    Description: Stop the gate that guard up started: SIGTERM, then SIGKILL after STOP_TIMEOUT.
    Parameters:
        files: the state files.
    Output: The stopped pid, or None when no Harpo gate was running. Removes gate.pid either way.
    """
    found = running_gate(files)
    files.pid.unlink(missing_ok=True)
    if found is None:
        return None
    pid = found[0]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return pid
        deadline = time.monotonic() + STOP_TIMEOUT
        while time.monotonic() < deadline:
            if _reaped(pid):
                return pid
            time.sleep(0.05)
    return pid


def gate_command(port: int, token_file: Path, audit_dir: Path) -> list[str]:
    """
    Description: The command that runs the gate: this interpreter, with the flags of the hook commands (harness.python_flags).
    Parameters:
        port: the port to listen on (0: any free port).
        token_file: the token file.
        audit_dir: where the gate writes the findings and proof logs (G6).
    Output: The argument list.
    """
    return [sys.executable, *harness.python_flags(), "-m", "Harpocrates.gate", "--port", str(port),
            "--token-file", str(token_file), "--audit-dir", str(audit_dir)]


def start_gate(files: Files, port: int, token: str) -> tuple[int, int]:
    """
    Description: Start the gate in the background, in its own session, and wait for it to listen.
    Parameters:
        files: the state files.
        port: the port to ask for (0: any free port).
        token: the gate token, written to gate.token first.
    Output: (pid, port it listens on). Raises RefusalError, after stopping the process, when the gate exits or does
        not print its listening line within START_TIMEOUT; the message is the last line of its log, or, when the log
        is empty, what happened instead (it exited, printed something else first, or never answered).
    """
    files.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    session_token.write_token_file(files.token, token)
    files.out.unlink(missing_ok=True)
    out_fd = os.open(files.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    log_fd = os.open(files.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(log_fd, 0o600)
        proc = subprocess.Popen(gate_command(port, files.token, files.state), stdin=subprocess.DEVNULL, stdout=out_fd, stderr=log_fd,
                                cwd=files.state, start_new_session=True, close_fds=True)
    finally:
        os.close(out_fd)
        os.close(log_fd)
    deadline = time.monotonic() + START_TIMEOUT
    why = f"it did not start listening within {START_TIMEOUT:.0f} s"
    while time.monotonic() < deadline and proc.poll() is None:
        first = files.out.read_text(encoding="utf-8", errors="replace").partition("\n")
        if first[1]:
            try:
                return proc.pid, int(json.loads(first[0])["listening"].rsplit(":", 1)[1])
            except (ValueError, KeyError, TypeError, AttributeError):
                why = f"its first output line was not its listening line: {first[0][:120]!r}"
                break
        time.sleep(0.05)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    elif not why.startswith("its first"):
        why = f"it exited with status {proc.returncode}"
    lines = files.log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    raise RefusalError(f"The gate did not start: {lines[-1] if lines else why}. Nothing was routed to it.")


# --- guard up / guard down -----------------------------------------------------------------------------------------

def _state(dry_run: bool, outcome: Outcome) -> Optional[tuple[Files, Optional[str], dict]]:
    """
    Description: Open the state folder and the install record, as `harpo install` does.
    Parameters:
        dry_run: look only (never migrate ~/.harpocrates).
        outcome: where a refusal is reported.
    Output: (files, record text, record), or None after reporting why not.
    """
    state = paths.peek_state_dir() if dry_run else paths.state_dir()
    try:
        paths.require_usable(state)
    except paths.StateDirUnusableError as exc:
        outcome.fail(str(exc))
        return None
    files = Files(state)
    text, record, problem = install._load_record(files.record)
    if problem:
        outcome.fail(f"Install record unreadable: {problem} Fix or delete {install._tilde(files.record)}, then re-run.")
        return None
    return files, text, record


def _write_record(files: Files, old: Optional[str], record: dict) -> None:
    """
    Description: Write the install record, or delete it when nothing is left in it.
    Parameters:
        files: the state files.
        old: its current text.
        record: the new record.
    Output: None.
    """
    keep = any(k in record for k in (*install.TARGETS, "gate"))
    text = json.dumps({"version": 1, **record}, indent=2) + "\n" if keep else None
    if text != old:
        install._write(install._Edit("install record", files.record, old, text, text is None, private=True))


def up(targets: Sequence[str], port: int = DEFAULT_PORT, dry_run: bool = False) -> Outcome:
    """
    Description: `harpo guard up`: start the gate with a new token and route each target harness to it.
    Parameters:
        targets: claude-code and/or codex.
        port: the port (0: any free port).
        dry_run: report what would change; start and write nothing.
    Output: The Outcome. Refused harnesses are reported and get nothing; with none left, nothing starts.
    """
    outcome = Outcome()
    opened = _state(dry_run, outcome)
    if opened is None:
        return outcome
    files, rtext, record = opened
    home = Path.home()
    ready = []
    for target in targets:
        rel, plan = PLANS[target]
        try:
            plan(home / rel, port or DEFAULT_PORT, "x")  # a dry plan: refusals do not depend on the port or token
            ready.append(target)
        except RefusalError as exc:
            outcome.fail(f"{install.NAMES[target]}: refused: {exc}")
    if not ready:
        return outcome.fail("No harness was routed to the gate.")
    if dry_run:
        outcome.lines.append(f"Dry run: would start the gate on 127.0.0.1:{port or 'any free port'} and route "
                             + ", ".join(install.NAMES[t] for t in ready) + " to it.")
        return outcome
    stopped = stop_gate(files)
    if stopped:
        outcome.lines.append(f"Stopped the earlier gate (pid {stopped}); its token no longer works.")
    token = session_token.new_token()
    try:
        pid, port = start_gate(files, port, token)
    except RefusalError as exc:
        return outcome.fail(str(exc))
    prior = (record.get("gate") or {}).get("routes") or {} if isinstance(record.get("gate"), dict) else {}
    routes: dict = {}
    try:
        for target in ready:
            routes[target] = _route(files, home, target, port, token, prior.get(target), outcome)
    except (RefusalError, OSError) as exc:
        stop_gate(files)
        return outcome.fail(f"Routing failed ({exc}); the gate was stopped, so the agents cannot reach a provider. "
                            "Run `harpo guard down` to restore their settings.")
    _private(files.pid, json.dumps({"pid": pid, "port": port}) + "\n")
    record["gate"] = {"enabled": True, "port": port, "token_header": TOKEN_HEADER, "routes": routes}
    _write_record(files, rtext, record)
    outcome.lines.insert(0, f"Gate listening on http://127.0.0.1:{port} (pid {pid}); request log {install._tilde(files.log)}.")
    outcome.lines.append("Restart agent sessions that were already running: they keep the old address and token.")
    return outcome


def _route(files: Files, home: Path, target: str, port: int, token: str, prior: Optional[dict],
           outcome: Outcome) -> dict:
    """
    Description: Write one harness's routing, keeping a snapshot of the file as it was without Harpo's gate lines.
    Parameters:
        files: the state files.
        home: the home folder.
        target: claude-code or codex.
        port: the gate's port.
        token: the gate token.
        prior: this target's route entry from the previous guard up, or None.
        outcome: where the step is reported.
    Output: The route entry for the install record: the file, whether it existed, and the hash of what was written.
    """
    rel, plan = PLANS[target]
    path = home / rel
    old, new = plan(path, port, token)
    snapshot = files.snapshots / target
    unchanged = prior is not None and _sha(old) == prior.get("written")
    if unchanged:  # still what the previous guard up wrote: its snapshot is still the file's true earlier text
        existed = bool(prior.get("existed"))
    else:
        base = None if old is None else plan(path, None)[1]
        existed = base is not None
        if existed:
            _private(snapshot, base)
        else:
            snapshot.unlink(missing_ok=True)
    edit = install._Edit(install.NAMES[target], path, old, new, False)
    install._write(edit)
    outcome.lines.append(f"{install.NAMES[target]}: routed to the gate ({install._tilde(path)})")
    return {"path": str(path), "existed": existed, "written": _sha(new)}


def down(dry_run: bool = False) -> Outcome:
    """
    Description: `harpo guard down`: take the routing out, drop the gate entry, then stop the gate.
    Parameters:
        dry_run: report what would change; write and stop nothing.
    Output: The Outcome.
    """
    outcome = Outcome()
    opened = _state(dry_run, outcome)
    if opened is None:
        return outcome
    files, rtext, record = opened
    gate = record.get("gate") if isinstance(record.get("gate"), dict) else {}
    routes = gate.get("routes") if isinstance(gate.get("routes"), dict) else {}
    for target, entry in routes.items():
        if target not in PLANS or not isinstance(entry, dict):
            continue
        path = Path(str(entry.get("path")))
        try:
            outcome.lines.append(_unroute(files, target, path, entry, dry_run))
        except (RefusalError, OSError) as exc:
            outcome.fail(f"{install.NAMES[target]}: could not restore {install._tilde(path)} ({exc}).")
    if dry_run:
        return outcome
    if not outcome.failed and "gate" in record:
        record.pop("gate")
        _write_record(files, rtext, record)
    stopped = stop_gate(files)
    files.token.unlink(missing_ok=True)
    files.out.unlink(missing_ok=True)
    outcome.lines.append(f"Stopped the gate (pid {stopped})." if stopped else "No gate was running.")
    return outcome


def _unroute(files: Files, target: str, path: Path, entry: dict, dry_run: bool) -> str:
    """
    Description: Restore one harness's file: its exact earlier text if nobody changed it since guard up, else the
        current text without Harpo's lines.
    Parameters:
        files: the state files.
        target: claude-code or codex.
        path: the routed file.
        entry: the route entry from the install record.
        dry_run: report only.
    Output: One report line.
    """
    name = install.NAMES[target]
    snapshot = files.snapshots / target
    cur = install._read(path)
    if cur is None:
        return f"{name}: {install._tilde(path)} is gone; nothing to restore."
    if _sha(cur) == entry.get("written"):
        new = snapshot.read_text(encoding="utf-8") if entry.get("existed") else None
        how = "restored exactly" if new is not None else "removed (guard up had created it)"
    else:
        new = PLANS[target][1](path, None)[1]
        how = "changed since guard up, so only Harpo's lines were removed"
    if not dry_run:
        if new != cur:
            install._write(install._Edit(name, path, cur, new, True, backup_first=False))
        snapshot.unlink(missing_ok=True)
        try:
            files.snapshots.rmdir()
        except OSError:
            pass
    return f"{name}: {install._tilde(path)} {how}" + (" (dry run)" if dry_run else "")


# --- harpo doctor --------------------------------------------------------------------------------------------------

def _probe(port: int, token: Optional[str]) -> tuple[int, bytes]:
    """
    Description: Send one request to the gate that it must refuse: no route is called that forwards anything.
    Parameters:
        port: the gate's port.
        token: the gate token, or None to send none.
    Output: (status, body). Raises OSError when nothing answers.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request("GET", "/harpo-doctor", headers={TOKEN_HEADER: token} if token else {})
        response = conn.getresponse()
        return response.status, response.read(4096)
    except http.client.HTTPException as exc:
        raise OSError(str(exc)) from None
    finally:
        conn.close()


def _routed(target: str, path: Path, port: int, token: str) -> Optional[str]:
    """
    Description: Check that a harness's config names the gate's port and the current token.
    Parameters:
        target: claude-code or codex.
        path: its config file.
        port: the gate's port.
        token: the current token.
    Output: None when it does, otherwise what is wrong.
    """
    text = install._read(path)
    if text is None:
        return f"{install._tilde(path)} is missing"
    if target == "claude-code":
        env = (json.loads(text) or {}).get("env") or {}
        if env.get("ANTHROPIC_BASE_URL") != f"http://127.0.0.1:{port}":
            return "ANTHROPIC_BASE_URL does not name the gate"
        if env.get("ANTHROPIC_CUSTOM_HEADERS") != f"{TOKEN_HEADER}: {token}":
            return "ANTHROPIC_CUSTOM_HEADERS does not carry the current gate token"
        return None
    data = install._parse_toml(text, path, "isn't valid TOML")
    provider = (data.get("model_providers") or {}).get(PROVIDER) or {}
    if data.get("model_provider") != PROVIDER or provider.get("base_url") != f"http://127.0.0.1:{port}/v1":
        return "model_provider does not route Codex to the gate"
    if (provider.get("http_headers") or {}).get(TOKEN_HEADER) != token:
        return "the provider's http_headers do not carry the current gate token"
    return None


def doctor(cwd: Path, environ: Optional[Mapping[str, str]] = None) -> list[Check]:
    """
    Description: `harpo doctor`: check, on this machine only, that the agents are routed through a working gate.
    Parameters:
        cwd: the folder whose project settings the SEC-10 check reads.
        environ: the environment Claude Code would run in (default: this process's).
    Output: The checks, in order; any "fail" means the agents are not protected as configured.
    """
    checks: list[Check] = []
    state = paths.peek_state_dir()
    if paths.unusable(state):
        return [Check("state folder", "fail", f"{install._tilde(state)} is a symlink or a file; move it away.")]
    files = Files(state)
    _text, record, problem = install._load_record(files.record)
    if problem:
        return [Check("install record", "fail", problem)]
    hooks = [install.NAMES[t] for t in install.TARGETS if isinstance(record.get(t), dict)]
    checks.append(Check("read hook", "ok" if hooks else "warn",
                        f"installed for {', '.join(hooks)}" if hooks else "not installed: run `harpo install`"))
    proof = audit.verify(state)
    checks.append(Check("proof log", "fail" if proof.problem else "ok",
                        f"{proof.problem}: see `harpo proof`" if proof.problem
                        else f"{len(proof.entries)} entries, the chain verifies"))
    try:
        entry = gatecheck.read_gate_entry()
    except gatecheck.GateRecordError as exc:
        return [*checks, Check("gate entry", "fail", str(exc))]
    gate = entry[0] if entry else None
    if gate is None:
        return [*checks, Check("gate entry", "fail", "the gate is not set up: run `harpo guard up`")]
    checks.append(Check("gate entry", "ok", f"port {gate.port}"))
    found = running_gate(files)
    if found is None or found[1] != gate.port:
        return [*checks, Check("gate process", "fail", "the gate is not running: run `harpo guard up`")]
    checks.append(Check("gate process", "ok", f"pid {found[0]}"))
    try:
        token = session_token.read_token_file(files.token)
        bare, signed = _probe(gate.port, None), _probe(gate.port, token)
    except (OSError, ValueError) as exc:
        return [*checks, Check("gate answers", "fail", f"{exc}: run `harpo guard up`")]
    ok = bare[0] == 401 and b"HARPO-BLOCK-TOKEN" in bare[1] and signed[0] == 400 and b"HARPO-BLOCK-ROUTE" in signed[1]
    checks.append(Check("gate answers", "ok" if ok else "fail",
                        "refuses a request without the token, accepts the token" if ok
                        else f"unexpected answers ({bare[0]}, {signed[0]}): another program may hold port {gate.port}"))
    routes = (record.get("gate") or {}).get("routes") or {}
    for target in install.TARGETS:
        if target not in routes:
            checks.append(Check(f"{install.NAMES[target]} routing", "warn", "not routed to the gate"))
            continue
        try:
            wrong = _routed(target, Path(str(routes[target].get("path"))), gate.port, token)
        except (RefusalError, ValueError, AttributeError) as exc:
            wrong = str(exc)
        checks.append(Check(f"{install.NAMES[target]} routing", "fail" if wrong else "ok",
                            f"{wrong}: run `harpo guard up`" if wrong else "routed to the gate with the current token"))
    findings = gatecheck.bypass_findings(cwd, environ)
    checks += [Check("bypass (SEC-10)", "fail", f.text) for f in findings]
    if not findings:
        checks.append(Check("bypass (SEC-10)", "ok", f"no setting in {install._tilde(cwd)} moves Claude Code off the gate"))
    return checks


def render_checks(checks: Sequence[Check]) -> str:
    """
    Description: Format doctor checks as aligned lines.
    Parameters:
        checks: the checks.
    Output: One line per check.
    """
    return "\n".join(f"{c.status.upper():<5} {c.name:<22} {c.detail}" for c in checks)
