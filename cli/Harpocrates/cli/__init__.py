"""
Command-line interface for Harpo secrets detection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, NoReturn, Optional, cast

import typer
from rich.console import Console
from rich.table import Table

from Harpocrates.core.result import ScanResult, Severity, findings_to_json
from Harpocrates.core.scanner import ScanEngine, scan_directory, scan_file, scan_git_history
from Harpocrates.paths import StateDirUnusableError

app = typer.Typer(
    name="harpo",
    help="Harpo - Secrets detection for code repositories",
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)

# Severity ranking (higher value = more severe). Used by --fail-on to
# determine which findings should cause a non-zero exit code.
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

# String tokens accepted by --fail-on. "none" disables the gate entirely
# (findings are reported but never cause a non-zero exit code).
_FAIL_ON_NONE = "none"
_VALID_FAIL_ON_LEVELS = (
    _FAIL_ON_NONE,
    Severity.INFO.value,
    Severity.LOW.value,
    Severity.MEDIUM.value,
    Severity.HIGH.value,
    Severity.CRITICAL.value,
)
_VALID_ENGINES = ("auto", "python", "rust")


def _resolve_fail_on(raw: str) -> Optional[Severity]:
    """
    Description: Parse the minimum severity used to gate scan exit status.
    Parameters:
        raw: User-supplied --fail-on value, normalized by trimming and lowercasing.
    Output: A Severity, or None when the value is none; raises typer.BadParameter for unsupported values.
    """
    normalized = raw.strip().lower()
    if normalized == _FAIL_ON_NONE:
        return None
    try:
        return Severity(normalized)
    except ValueError as exc:
        valid = ", ".join(_VALID_FAIL_ON_LEVELS)
        raise typer.BadParameter(
            f"Invalid --fail-on value {raw!r}. Must be one of: {valid}"
        ) from exc


def _fail_on_callback(value: str) -> str:
    """
    Description: Validate --fail-on during argument parsing so invalid input fails before scanning.
    Parameters:
        value: User-supplied severity threshold text.
    Output: Original value after validation; raises typer.BadParameter for invalid thresholds.
    """
    _resolve_fail_on(value)
    return value


def _engine_callback(value: str) -> str:
    """
    Description: Validate and normalize the scan-engine choice during argument parsing.
    Parameters:
        value: User-supplied engine name.
    Output: The trimmed, lowercased engine: auto, python, or rust; raises typer.BadParameter for unsupported names.
    """
    normalized = value.strip().lower()
    if normalized not in _VALID_ENGINES:
        valid = ", ".join(_VALID_ENGINES)
        raise typer.BadParameter(f"Invalid --engine value {value!r}. Must be one of: {valid}")
    return normalized


def _should_fail(findings, min_severity: Optional[Severity]) -> bool:
    """
    Description: Determine whether findings meet the configured minimum severity for a failing exit status.
    Parameters:
        findings: Iterable of findings with severity values.
        min_severity: Minimum failing severity, or None to disable the gate.
    Output: True if any finding meets or exceeds the minimum severity, otherwise False; unknown severity
        keys raise KeyError.
    """
    if min_severity is None:
        return False
    threshold = _SEVERITY_RANK[min_severity]
    return any(_SEVERITY_RANK[f.severity] >= threshold for f in findings)


def _version_callback(value: bool) -> None:
    """
    Description: Handle the eager version flag without running a scan.
    Parameters:
        value: Whether --version was requested.
    Output: None when false; otherwise prints the package version and raises typer.Exit with code 0.
    """
    if value:
        from Harpocrates import __version__

        typer.echo(f"harpo {__version__}")
        raise typer.Exit(code=0)


@app.callback(help=(
    'Harpo - Secrets detection for code repositories'
))
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """
    Description: Provide the root CLI callback for Harpo secret detection.
    Parameters:
        version: Eager version flag handled by _version_callback.
    Output: None; the callback body has no side effects.
    """


_SEVERITY_STYLES = {
    Severity.CRITICAL: "red bold",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "blue",
    Severity.INFO: "white",
}


def _scan_paths(paths: List[Path], *, history: bool, recursive: bool, memory_cap: int, ignore_patterns: set,
                verifier: object, ml_threshold: float, engine: ScanEngine) -> ScanResult:
    """
    Description: Scan each path given to `harpo scan` (a file, a folder, or its git history) and add up the results.
    Parameters:
        paths: the paths.
        history: scan each path's git history instead of its files.
        recursive: descend into subfolders.
        memory_cap: the most bytes one file's scan may hold (FR-CORE-05).
        ignore_patterns: extra ignore patterns for folder scans.
        verifier: the loaded model.
        ml_threshold: the least model confidence kept.
        engine: auto, rust or python.
    Output: One ScanResult over every path. A missing path, or a file that cannot be read, is skipped with a warning
        on stderr; a history error exits 2.
    """
    all_findings = []
    total_files = 0
    total_lines = 0
    total_duration = 0.0
    all_errors: list = []

    for path in paths:
        if not path.exists():
            error_console.print(f"[yellow]⚠[/yellow] Path not found, skipping: {path}")
            continue

        if history:
            try:
                r = scan_git_history(path)
            except ValueError as e:
                error_console.print(f"[red]✗[/red] {e}")
                raise typer.Exit(code=2) from e
        elif path.is_dir():
            r = scan_directory(
                path,
                recursive=recursive,
                memory_cap=memory_cap,
                ignore_patterns=ignore_patterns,
                verifier=verifier,
                ml_threshold=ml_threshold,
                engine=engine,
            )
        else:
            try:
                r = scan_file(
                    path,
                    memory_cap=memory_cap,
                    verifier=verifier,
                    ml_threshold=ml_threshold,
                    engine=engine,
                )
            except UnicodeDecodeError:
                error_console.print(
                    f"[yellow]⚠[/yellow] Skipping binary/unreadable file: {path}"
                )
                continue
            except (OSError, PermissionError) as e:
                error_console.print(
                    f"[yellow]⚠[/yellow] Skipping unreadable file {path}: {e}"
                )
                continue

        all_findings.extend(r.findings)
        total_files += r.scanned_files
        total_lines += r.total_lines
        total_duration += r.duration_ms
        if r.errors:
            all_errors.extend(r.errors)

    return ScanResult(
        findings=all_findings,
        scanned_files=total_files,
        total_lines=total_lines,
        duration_ms=total_duration,
        errors=all_errors,
    )


def _print_findings_table(result: ScanResult, show_secrets: bool) -> None:
    """
    Description: Print scan findings as a table, most severe first, then the totals.
    Parameters:
        result: the scan result, with at least one finding.
        show_secrets: print full tokens instead of redacted ones.
    Output: None; prints to the console.
    """
    table = Table(
        title=f"Found {len(result.findings)} potential secrets",
        show_header=True
    )
    table.add_column("Severity", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Location")
    table.add_column("Secret")

    for finding in sorted(
        result.findings,
        key=lambda f: (f.severity.value, f.file or "", f.line or 0)
    ):
        severity_style = _SEVERITY_STYLES.get(finding.severity, "white")
        severity_text = f"[{severity_style}]{finding.severity.value.upper()}[/{severity_style}]"

        # Format location
        location = f"{finding.file}:{finding.line}" if finding.file else "text input"

        # Redacted by default; full token only when --show-secrets is set
        if show_secrets:
            secret_cell = finding.token or finding.snippet or "—"
        else:
            secret_cell = finding.redacted_token or "—"

        table.add_row(
            severity_text,
            finding.type,
            location,
            secret_cell,
        )

    console.print(table)
    console.print(
        f"\nScanned {result.scanned_files} files ({result.total_lines} lines) "
        f"in {result.duration_ms:.0f}ms"
    )
    console.print(
        f"[yellow]⚠[/yellow]  Found {result.critical_count} critical, "
        f"{result.high_count} high severity secrets"
    )


@app.command(help=(
    '\n'
    '    Scan a file or directory for secrets.\n'
    '\n'
    '    By default, detected secret tokens are redacted in both table and JSON\n'
    '    output. Pass --show-secrets to display full tokens (use with caution).\n'
    '\n'
    '    Examples:\n'
    '\n'
    '        # Scan a single file\n'
    '        harpo scan config.env\n'
    '\n'
    '        # Scan a directory recursively\n'
    '        harpo scan ./my_project\n'
    '\n'
    '        # Output as JSON\n'
    '        harpo scan ./my_project --json\n'
    '\n'
    '        # Ignore specific patterns\n'
    '        harpo scan ./my_project --ignore "*.test.js,test_*"\n'
    '\n'
    '        # Fewer, higher-confidence findings\n'
    '        harpo scan ./my_project --ml-threshold 0.7\n'
    '\n'
    '        # Require the native regex/entropy scanner; ML still runs in Python\n'
    '        harpo scan ./my_project --engine rust\n'
    '\n'
    '        # Display full token values (NOT recommended outside local debugging)\n'
    '        harpo scan config.env --show-secrets\n'
    '\n'
    '        # Scan every commit in git history (finds secrets deleted from disk)\n'
    '        harpo scan . --history\n'
    '\n'
    '        # Only fail CI on high or critical findings\n'
    '        harpo scan ./my_project --fail-on high\n'
    '\n'
    '        # Report findings but never return a non-zero exit code\n'
    '        harpo scan ./my_project --fail-on none\n'
    '\n'
    '        # XAI: emit JSON with per-feature SHAP contributions\n'
    '        harpo scan ./my_project --explain | jq '
    "'.findings[0].explanation.top_positive'\n"
    '    '
))
def scan(
    paths: Optional[List[Path]] = typer.Argument(default=None, help="Files or directories to scan"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r",
        help="Scan directories recursively"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    max_file_size: int = typer.Option(64, "--max-size", min=1, help=(
        "Memory cap per file in MB. Every file is scanned whole, in pieces, whatever its size.")),
    engine: str = typer.Option(
        "auto",
        "--engine",
        help="Scanning engine: auto, python, or rust. Auto prefers a built Rust binary.",
        callback=_engine_callback,
    ),
    ignore: Optional[str] = typer.Option(
        None, "--ignore",
        help="Comma-separated patterns to ignore"
    ),
    ml_threshold: float = typer.Option(
        0.19, "--ml-threshold",
        help=(
            "Extra floor on combined confidence (0.0-1.0, default: 0.19). "
            "The model's own commit threshold in model_config.json decides "
            "first; at the default this floor never removes a finding."
        ),
    ),
    show_secrets: bool = typer.Option(
        False, "--show-secrets",
        help="Display full secret tokens instead of redacted versions"
    ),
    fail_on: str = typer.Option(
        Severity.MEDIUM.value, "--fail-on",
        help=(
            "Minimum severity that causes a non-zero exit code. One of: "
            "critical, high, medium, low, info, none. Default: medium."
        ),
        callback=_fail_on_callback,
    ),
    history: bool = typer.Option(
        False, "--history", help="Scan lines added in every git commit instead of files on disk"
    ),
    explain: bool = typer.Option(
        False, "--explain",
        help=(
            "Emit JSON with per-feature ML contribution scores (TreeSHAP). "
            "Loads xgboost lazily — cost paid only when set."
        ),
    ),
) -> None:
    """
    Description: Scan supplied files, directories, or Git history and report findings with a configurable
        severity exit gate. History mode also finds secrets that were deleted from disk.
    Parameters:
        paths: Paths to scan; no paths exits successfully without scanning.
        recursive: Whether directory scans include subdirectories.
        json_output: Whether to print structured JSON instead of a table.
        max_file_size: The memory cap per file in MiB (FR-CORE-05), converted to bytes for scans.
        engine: Selected auto, python, or rust scan engine.
        ignore: Optional comma-separated ignore patterns.
        ml_threshold: Additional ML confidence floor between zero and one.
        show_secrets: Whether to include full token values; use with caution.
        fail_on: Minimum failing severity or none to disable finding-based failure.
        history: Whether to scan added lines across Git history instead of disk files.
        explain: Whether to output JSON with per-feature TreeSHAP contributions.
    Output: No return on handled paths; prints reports and diagnostics and raises typer.Exit with 0 for
        success, 1 for threshold findings, or 2 for handled configuration, model, or scan errors. Loads the
        shipped model eagerly and fails closed if loading fails; scans input paths and may load explanation
        dependencies. Unhandled scanner or explanation errors propagate. Detected tokens are redacted by
        default in table and JSON output. With the rust engine, ML still runs in Python.
    """
    # Zero-file early return — pre-commit passes no files when nothing is staged.
    if not paths:
        raise typer.Exit(code=0)

    if not (0.0 <= ml_threshold <= 1.0):
        error_console.print(
            f"[red]✗[/red] --ml-threshold must be between 0.0 and 1.0, got: {ml_threshold}"
        )
        raise typer.Exit(code=2)

    if engine == "rust":
        from Harpocrates.core.rust_backend import RustScannerBackend, RustScannerError

        try:
            RustScannerBackend.discover(required=True)
        except RustScannerError as exc:
            error_console.print(f"[red]✗[/red] {exc}")
            raise typer.Exit(code=2) from exc

    # Already validated by _fail_on_callback at parse time; safe to call.
    fail_on_severity = _resolve_fail_on(fail_on)

    ignore_patterns = set(ignore.split(",")) if ignore else set()

    memory_cap = max_file_size * 1024 * 1024

    # The shipped model always runs. Fail closed: without it, regex and entropy alone would pass
    # secrets the model is there to catch.
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    try:
        verifier = OnnxVerifier(layer="commit", lazy_load=False)
    except Exception as e:
        error_console.print(f"[red]✗[/red] ML model could not be loaded: {e}")
        raise typer.Exit(code=2) from e

    result = _scan_paths(paths, history=history, recursive=recursive, memory_cap=memory_cap,
                         ignore_patterns=ignore_patterns, verifier=verifier, ml_threshold=ml_threshold,
                         engine=cast(ScanEngine, engine))

    # Handle errors
    if result.errors:
        for error in result.errors:
            error_console.print(f"[yellow]⚠[/yellow]  {error}")
        fatal_errors = [
            error
            for error in result.errors
            if "; used Python fallback:" not in error
        ]
        if fatal_errors:
            raise typer.Exit(code=2)

    exit_on_findings = (
        1 if _should_fail(result.findings, fail_on_severity) else 0
    )

    if explain:
        # Lazy import — xgboost is never loaded on the default scan path.
        try:
            from Harpocrates.ml.context import extract_context_from_finding
            from Harpocrates.ml.explain import explain_finding
        except ImportError:
            error_console.print("[red]✗[/red] --explain requires xgboost: pip install 'harpo-scan[explain]'")
            raise typer.Exit(code=2)

        payload = []
        shown = findings_to_json(result.findings, include_token=show_secrets)
        for f, finding_json in zip(result.findings, shown):
            ctx = extract_context_from_finding(f)
            explanation = explain_finding(f, ctx)
            payload.append({
                "finding": finding_json,
                "explanation": explanation.to_dict() if explanation else None,
            })
        print(json.dumps({"findings": payload}, indent=2))
        raise typer.Exit(code=exit_on_findings)

    if json_output:
        print(json.dumps(result.to_dict(include_token=show_secrets), indent=2))
        raise typer.Exit(code=exit_on_findings)

    if not result.found_secrets:
        console.print("[green]✓[/green] No secrets detected")
        console.print(
            f"Scanned {result.scanned_files} files ({result.total_lines} lines) "
            f"in {result.duration_ms:.0f}ms"
        )
        raise typer.Exit(code=0)

    _print_findings_table(result, show_secrets)

    raise typer.Exit(code=exit_on_findings)


@app.command(help=(
    'Display version information.'
))
def version() -> None:
    """
    Description: Display the installed Harpo distribution version.
    Parameters: None
    Output: None; prints version information and propagates metadata lookup or output errors.
    """
    from importlib.metadata import version as _version
    console.print(f"Harpo version {_version('harpo-scan')}")


@app.command(help=(
    "Show each detected agent harness's support tier, or print one harness's setup."
))
def setup(
    harness: Optional[str] = typer.Argument(
        None, help="claude-code or codex: print its read-tool setup. Omit to list detected harnesses."
    ),
) -> None:
    """
    Description: Display detected harness support tiers or a selected harness setup guide.
    Parameters:
        harness: Optional claude-code or codex guide identifier; None lists detected harnesses.
    Output: None after printing the report or guide; prints unsupported-guide errors and raises
        typer.Exit with code 2. Detection or output errors may propagate.
    """
    from Harpocrates.harness import detect, setup_instructions, tier_report

    if harness is None:
        typer.echo(tier_report(detect()))
        return
    try:
        typer.echo(setup_instructions(harness))  # plain echo: rich would eat [toml] tables as markup
    except ValueError as exc:
        typer.echo(f"✗ {exc}", err=True)  # plain: the message echoes user input
        raise typer.Exit(code=2) from exc


def _install_run(remove: bool, claude_code: bool, codex: bool, dry_run: bool) -> None:
    """
    Description: Choose explicit or detected harness targets and run installation or removal with a
        rendered report.
    Parameters:
        remove: Whether to uninstall instead of install.
        claude_code: Whether Claude Code was explicitly selected.
        codex: Whether Codex CLI was explicitly selected.
        dry_run: Whether to preview changes.
    Output: None on success; prints reports and raises typer.Exit with code 1 for no detected target
        or failed reports. May write configuration, backups, and ownership state or invoke Claude
        through installation helpers; unhandled helper errors propagate.
    """
    from Harpocrates import install as _install

    targets = [t for t, wanted in (("claude-code", claude_code), ("codex", codex)) if wanted]
    if not targets:
        targets = _install.detected_targets()
        if not targets:
            typer.echo("No Claude Code or Codex CLI detected. Name one: --claude-code or --codex.", err=True)
            raise typer.Exit(code=1)
    reports = (_install.uninstall_harnesses if remove else _install.install_harnesses)(targets, dry_run)
    typer.echo(_install.render(reports, dry_run, remove))  # plain echo: rich would eat [brackets] as markup
    if _install.failed(reports):
        raise typer.Exit(code=1)


@app.command("install", help=(
    "Write the read-tool setup into each harness's own config "
    '(FR-INSTALL-01). No flag: every detected harness.'
))
def install_cmd(
    claude_code: bool = typer.Option(False, "--claude-code", help="Install for Claude Code."),
    codex: bool = typer.Option(False, "--codex", help="Install for Codex CLI."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change; write and run nothing."),
) -> None:
    """
    Description: Install the read-tool setup into the selected or detected harness configuration
        (FR-INSTALL-01). With no harness flag, every detected harness is used.
    Parameters:
        claude_code: Whether to explicitly select Claude Code.
        codex: Whether to explicitly select Codex CLI.
        dry_run: Whether to show proposed changes without applying them.
    Output: None on success; prints reports, may update harness files and MCP registration, and raises
        typer.Exit with code 1 for missing targets or failed reports. Installation errors may propagate.
    """
    _install_run(False, claude_code, codex, dry_run)


@app.command("uninstall", help=(
    'Remove exactly what `harpo install` added; everything else and '
    'the backups stay.'
))
def uninstall_cmd(
    claude_code: bool = typer.Option(False, "--claude-code", help="Uninstall from Claude Code."),
    codex: bool = typer.Option(False, "--codex", help="Uninstall from Codex CLI."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change; write and run nothing."),
) -> None:
    """
    Description: Remove configuration owned by Harpo installation while retaining other settings and
        backups.
    Parameters:
        claude_code: Whether to explicitly select Claude Code.
        codex: Whether to explicitly select Codex CLI.
        dry_run: Whether to show proposed removal without applying it.
    Output: None on success; prints reports, may remove owned configuration and MCP registration, and
        raises typer.Exit with code 1 for missing targets or failed reports. Removal errors may propagate.
    """
    _install_run(True, claude_code, codex, dry_run)


def _interactive() -> bool:
    """
    Description: Check whether standard input can support an interactive confirmation prompt.
    Parameters: None
    Output: True if stdin is a terminal, otherwise False; stream errors propagate.
    """
    return sys.stdin.isatty()


@app.command(help=(
    'Let agents read a path that Harpo would otherwise flag '
    '(FR-READ-05). Every use is logged.\n'
    '\n'
    '    Exit code 0 ok, 1 refused or error, 3 a broad name needs '
    'confirmation (--yes).\n'
    '    '
))
def allow(
    path: Optional[Path] = typer.Argument(None, help="File or folder the agent may read even if it holds secrets."),
    name: Optional[str] = typer.Option(
        None, "--name", help="A folder or file name, allowed in every project (for example graphify-out)."),
    remove: bool = typer.Option(False, "--remove", help="Take PATH, or --name NAME, off the allowlist."),
    list_: bool = typer.Option(False, "--list", help="Show the allowlist."),
    yes: bool = typer.Option(False, "--yes", help="Confirm a broad name (src, lib, home, ...) without asking."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable result on stdout, for plugins."),
) -> None:
    """
    Description: Manage explicit path or name overrides that let agents read files Harpo would otherwise
        flag (FR-READ-05).
    Parameters:
        path: File or folder path to add or remove.
        name: Exact file or folder name to allow across projects, instead of a path.
        remove: Whether to remove the specified rule.
        list_: Whether to display existing rules.
        yes: Whether to confirm a broad name without prompting.
        as_json: Whether to emit machine-readable JSON and never prompt.
    Output: No return on handled paths; prints results and raises typer.Exit with 0 for success, 1 for
        refusal or handled error, or 3 when a broad name requires confirmation (pass --yes). May update the
        allowlist or prompt interactively; uses are logged by the hook. Unhandled filesystem or confirmation
        errors propagate.
    """
    from Harpocrates import allowlist

    def done(payload: dict, text: str, code: int = 0) -> NoReturn:
        """
        Description: Emit an allow-command result in the selected format and exit with its status.
        Parameters:
            payload: Dictionary serialized in JSON mode.
            text: Human-readable result used outside JSON mode.
            code: Exit status, defaulting to zero.
        Output: No return; writes stdout or, for non-JSON failures, stderr, then raises typer.Exit
            with code. Serialization and output errors propagate.
        """
        typer.echo(json.dumps(payload) if as_json else text, err=code != 0 and not as_json)
        raise typer.Exit(code=code)

    def fail(message: str, code: int = 1, needs_confirmation: bool = False) -> NoReturn:
        """
        Description: Build a structured allow-command failure and terminate with the selected
            status.
        Parameters:
            message: Failure explanation.
            code: Exit status, defaulting to one.
            needs_confirmation: Whether the client must request confirmation for a broad name.
        Output: No return; emits an error payload or text through done and raises typer.Exit with
            code. Output errors propagate.
        """
        done({"ok": False, "error": message, "needs_confirmation": needs_confirmation}, f"✗ {message}", code)

    logged = f"Uses are logged to {allowlist.log_file()}."
    if path is not None and name is not None:
        fail("Give a PATH or --name NAME, not both.")
    if list_ or (path is None and name is None and not remove):
        if path is not None or name is not None or remove:
            fail("--list takes no PATH, --name or --remove.")
        rows = allowlist.listing()
        shown = "\n".join(  # printable: a file name with a newline or escape can't fake lines
            allowlist.printable(v) if k == "path" else f"name:{v}" for k, v in rows) or "(empty)"
        done({"entries": [{"kind": k, "value": v} for k, v in rows]},
             f"{shown}\nRemove one with --remove PATH or --remove --name NAME. {logged}")
    if path is None and name is None:
        fail("--remove needs a PATH or --name NAME.")
    kind = "name" if name is not None else "path"

    if remove:
        try:
            value = allowlist.remove_name(name) if name is not None else str(allowlist.remove(cast(Path, path)))
        except ValueError as exc:
            fail(str(exc))
        what = f"the name {value}" if kind == "name" else allowlist.printable(value)
        done({"ok": True, "action": "removed", "kind": kind, "value": value},
             f"Removed {what}. Earlier uses stay in {allowlist.log_file()}.")

    if name is not None:
        existed = name in allowlist.names()
        if not existed:
            try:
                allowlist.add_name(name, confirmed=yes)
            except allowlist.NeedsConfirmationError as exc:
                # --json never prompts: a plugin can't answer, so it gets exit 3 and decides what to do.
                if as_json or not _interactive():
                    fail(f"{exc}. Re-run with --yes to allow it.", 3, True)
                if not typer.confirm(f"{exc}. Allow it anyway?", default=False):
                    fail("Not allowed.")
                try:
                    allowlist.add_name(name, confirmed=True)
                except ValueError as exc2:
                    fail(str(exc2))
            except ValueError as exc:
                fail(str(exc))
        value, what = name, f"the name {name} in every project"
    else:
        target = cast(Path, path)
        existed = target.resolve() in allowlist.entries()
        try:
            value = str(allowlist.add(target))
        except StateDirUnusableError as exc:
            fail(str(exc))
        except OSError as exc:
            fail(f"{target}: {exc.strerror or 'not found'}")
        what = allowlist.printable(value)
    done({"ok": True, "action": "exists" if existed else "added", "kind": kind, "value": value},
         f"{'Already allowed' if existed else 'Allowed'} {what}. {logged}")


guard_app = typer.Typer(help="Route Claude Code and Codex through the egress gate, or take them off it.",
                        no_args_is_help=True)
app.add_typer(guard_app, name="guard")


@guard_app.command("up", help=(
    "Start the gate in the background with a new token and route each harness to it. "
    "No flag: every detected harness."
))
def guard_up(
    port: int = typer.Option(47321, "--port", help="Port on 127.0.0.1 (0: any free port)."),
    claude_code: bool = typer.Option(False, "--claude-code", help="Route Claude Code."),
    codex: bool = typer.Option(False, "--codex", help="Route Codex CLI."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change; start and write nothing."),
) -> None:
    """
    Description: `harpo guard up` (G7, FR-UX-01).
    Parameters:
        port: the gate's port.
        claude_code: route Claude Code.
        codex: route Codex CLI.
        dry_run: report only.
    Output: None; prints the outcome and exits 1 if anything was refused or failed.
    """
    from Harpocrates import guard
    from Harpocrates import install as _install

    targets = [t for t, wanted in (("claude-code", claude_code), ("codex", codex)) if wanted]
    targets = targets or _install.detected_targets()
    if not targets:
        typer.echo("No Claude Code or Codex CLI detected. Name one: --claude-code or --codex.", err=True)
        raise typer.Exit(code=1)
    _print_outcome(guard.up(targets, port, dry_run))


@guard_app.command("down", help="Take the agents off the gate, restore their settings, and stop the gate.")
def guard_down(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change; write and stop nothing."),
) -> None:
    """
    Description: `harpo guard down` (G7, FR-INSTALL-08).
    Parameters:
        dry_run: report only.
    Output: None; prints the outcome and exits 1 if a file could not be restored.
    """
    from Harpocrates import guard

    _print_outcome(guard.down(dry_run))


def _print_outcome(outcome: object) -> None:
    """
    Description: Print a guard Outcome and exit 1 if it failed.
    Parameters:
        outcome: a Harpocrates.guard.Outcome.
    Output: None, or typer.Exit(1).
    """
    typer.echo("\n".join(outcome.lines))  # type: ignore[attr-defined]  # plain echo: rich would eat [brackets]
    if outcome.failed:  # type: ignore[attr-defined]
        raise typer.Exit(code=1)


@app.command("doctor", help=(
    "Check on this machine that the agents are routed through a running gate with the current token, "
    "and that no setting bypasses it. Sends nothing to a provider."
))
def doctor_cmd(
    json_output: bool = typer.Option(False, "--json", help="Print the checks as JSON."),
) -> None:
    """
    Description: `harpo doctor` (G7, FR-INSTALL-02, FR-PROOF-05).
    Parameters:
        json_output: print JSON instead of lines.
    Output: None; exits 1 if any check fails.
    """
    from dataclasses import asdict

    from Harpocrates import guard

    checks = guard.doctor(Path.cwd())
    if json_output:
        typer.echo(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        typer.echo(guard.render_checks(checks))
    if any(c.status == "fail" for c in checks):
        raise typer.Exit(code=1)


@app.command("proof", help=(
    "Show the last requests the gate forwarded, as hashes and placeholder counts, and verify the proof chain. "
    "No request body is stored."
))
def proof_cmd(
    last: int = typer.Option(10, "--last", min=1, help="How many of the newest entries to show."),
    json_output: bool = typer.Option(False, "--json", help="Print the entries and the result as JSON."),
) -> None:
    """
    Description: `harpo proof` (G6, FR-PROOF-03).
    Parameters:
        last: how many entries to show.
        json_output: print JSON instead of lines.
    Output: None; exits 1 when the chain does not verify or the state folder is unusable.
    """
    from Harpocrates import audit, paths

    state = paths.peek_state_dir()
    if paths.unusable(state):
        typer.echo(f"Error: {state} is a symlink or a file; move it away.", err=True)
        raise typer.Exit(code=1)
    result = audit.verify(state)
    shown = result.entries[-last:]
    if json_output:
        typer.echo(json.dumps({"entries": shown, "total": len(result.entries), "verified": result.problem is None,
                               "problem": result.problem}, indent=2))
    else:
        for e in shown:
            counts = ", ".join(f"{kind} {n}" for kind, n in e["placeholders"].items()) or "no placeholders"
            typer.echo(f"{e['seq']:>6}  {e['time']}  {e['route']:<26} {e['bytes']:>9} B  sha256 {e['sha256'][:16]}  "
                       f"{counts}")
        typer.echo(f"Chain does not verify: {result.problem}" if result.problem
                   else f"Chain verifies: {len(result.entries)} entries.")
    if result.problem:
        raise typer.Exit(code=1)


def main() -> None:
    """
    Description: Run the Typer application as the CLI entry point.
    Parameters: None
    Output: None if the application returns; dispatches commands, prints CLI output, and may raise
        SystemExit or command errors.
    """
    app()


if __name__ == "__main__":
    main()
