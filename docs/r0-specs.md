# R0 specs: remaining work

Oct 18, 2026 · status: **awaiting Josh's approval.** One spec per component, a page or less each. Nothing below is implemented yet.

## 1. FR-CORE-07 part 2: Java method regex hang

**Problem.** `ml/features.py` (`_extract_context_features`, about line 1456) uses `^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(` with `re.M`. The three adjacent `\s*` groups with optional words between them backtrack cubically on runs of whitespace. Measured: 11 s on 2,000 spaces for the regex alone, and about 173 s for `extract_features` on a 5,000-space line.

**Input:** `full_text`, the context window around a candidate.

**Output:** `has_func_def` (bool), one model feature.

**Change.** Remove the ambiguity so each character can be matched only one way: `^[ \t]*(?:(?:public|private|protected)[ \t]+)?(?:static[ \t]+)?\w+[ \t]+\w+[ \t]*\(`. That is linear time. `requires-python` allows 3.10, so atomic groups (3.11+) are out.

**Equivalence risk.** `\s` also matches newlines, so the old pattern could match across lines, for example a type on one line and a name on the next. The new one can't. This is the one place the feature can change.

**Failure behavior.** If any snapshot file differs (`bench/results/r0-before/tools/run_all.sh`), the fix is not applied as is. Two fallbacks, both your call:
1. Keep `\s` but make it unambiguous: `^\s*(?:(?:public|private|protected)\s+)?(?:static\s+)?\w+\s+\w+\s*\(`. Its worst case needs measuring.
2. Bound the input: scan only lines up to N characters for this feature.

Either fallback, if it changes features, means retraining, so it waits for M6.

**Test plan, tests first:**
- `test_fr_core_07_java_func_pattern_whitespace_run`: 5,000 and 50,000 spaces plus `;var k="tok";` complete within 0.5 s.
- `test_fr_core_07_java_func_pattern_same_results`: a fixture list of Java, Python and JavaScript snippets gives the same `has_func_def` before and after, including multi-line signatures.
- Run `run_all.sh`: byte-identical.

## 2. DOC-DOCSTRING-01: docstring checker and rollout

**Script.** `scripts/check_docstrings.py`, tracked: the one tracked file under `scripts/`.
- **Input:** paths, files or directories, and an optional `--enforce` list.
- **Rule:** every `FunctionDef`, `AsyncFunctionDef` and `ClassDef` in the given modules has a docstring containing `Description:`, `Parameters:` and `Output:`, in that order, each at the start of a line. Tests are exempt. So are nested functions? **Proposed: no.** Nested functions are exempt only if they start with `_`. Your call.
- **Output:** `path:line: name: missing <heading>` per violation, exit 1 if any, exit 0 otherwise.
- **Failure behavior:** a file that doesn't parse is reported and the run fails. It never silently passes.

**Enforcement.**
- `pyproject.toml` gets `[tool.harpo.docstrings] enforce = [...]`, the list of modules converted so far.
- CI runs `python scripts/check_docstrings.py --enforce`.
- Each module commit adds that module to the list. When every module is in, the list becomes `["cli/Harpocrates"]`.

**Rollout.** 37 modules, 423 definitions, 1 already compliant. One module per commit, docstrings only, no code changes. FR-CLEAN-02's byte-identical findings check runs after every 10 commits and at the end. The work is mechanical, so the proposal is the `codex-personal` worker in batches of about 5 modules, with me reviewing every commit. Security modules (`hooks.py`, `allowlist.py`, `grants.py`, `harness.py`, `read.py`, `mcp/`) are docstring-only too, with no code changes, so they don't need FR-CLEAN-03 sign-off.

**Test plan:**
- `test_doc_docstring_01_missing_heading_fails`
- `test_doc_docstring_01_wrong_order_fails`
- `test_doc_docstring_01_compliant_passes`
- `test_doc_docstring_01_tests_exempt`
- `test_doc_docstring_01_unparseable_file_fails`

## 3. FR-NAME-01 to 07: rename to Harpo

**Decided:** the distribution is `harpo-scan` (FR-NAME-03).

**Open:** the import name. Behavox's `harpo` is one 33.8 KB wheel (`harpo-1.4.2-py3-none-any.whl`). Reading its import name means downloading it, and that needs your OK. PyPI's inspector blocks automated reads. **Proposal:** keep the import package `Harpocrates`. An import rename touches 64 files, gains nothing users see, and risks a collision. If you'd rather rename it, `harpo_scan` is the only name that can't collide.

**Scope:**

| ID | Change | Notes |
| --- | --- | --- |
| FR-NAME-01 | User-facing text says Harpo: README, docs, CLI help and errors, MCP server name `FastMCP("harpo")` | The MCP tool names `safe_read` and `safe_grep` don't change, so the installed agent deny and allow rules keep working |
| FR-NAME-02 | Console scripts `harpo` and `harpo-mcp`. `harpocrates` and `harpocrates-mcp` stay, print one deprecation line to stderr, then run the same entry point | Both run the same CLI test suite |
| FR-NAME-03 | Distribution `harpo-scan`. `harpocrates` on PyPI becomes a shim that depends on `harpo-scan` | Recorded in `design.md` |
| FR-NAME-04 | `~/.harpo`. On first run, copy `~/.harpocrates` there (copy, not move) and leave the original in place. Env vars `HARPO_*` first, then `HARPOCRATES_*` (4 today: `_DIR`, `_REQUIRE_RUST`, `_RUST_SCANNER`, `_RUST_TIMEOUT_SECONDS`) | ⚠ security, see below |
| FR-NAME-05 | `.pre-commit-hooks.yaml` gets `id: harpo`, and keeps `id: harpocrates` | Both run in a fixture repo |
| FR-NAME-06 | The placeholder `<<HARPO:type:xxxx>>` doesn't change | Existing tests pass unchanged |
| FR-NAME-07 | GitHub repo renamed last, after the release | Link check over every URL Harpo prints |

**⚠ Security for FR-NAME-04 (FR-CLEAN-03 sign-off needed):**
- **Guard both directories.** The guard in `allowlist.py`, including the obfuscated spellings it handles today (`.h\arpocrates`, `{.,}harpocrates`, `$'\056…'`), must cover `~/.harpo` and `~/.harpocrates`. Otherwise an agent can write the old allow file before migration runs, and the migration copies it into the new directory.
- **Deny rules for both.** The installer's Claude Code deny rules add `~/.harpo` and keep `~/.harpocrates`.
- **Migration fails closed.** It copies only when `~/.harpo` doesn't exist yet. It refuses symlinked sources. It writes to a temp directory and renames it into place. If the copy fails, Harpo keeps reading `~/.harpocrates` and never starts with an empty allowlist that has silently lost grants.
- **The env var check is a security check too.** `HARPO_DIR` set to a directory the agent can write would bypass the guard just as `HARPOCRATES_DIR` would today. The guard must apply to both names.

**Test plan:**
- `test_fr_name_01_no_user_facing_harpocrates`: a text search with a changelog and migration-notes allowlist.
- `test_fr_name_02_old_commands_warn_and_work`
- `test_fr_name_04_migration_copies_and_keeps_original`
- `test_fr_name_04_agent_cannot_edit_either_dir`
- `test_fr_name_04_symlinked_source_refused`
- `test_fr_name_04_env_precedence`
- `test_fr_name_05_both_hook_ids`
- Run `run_all.sh`: byte-identical.

**Order:** FR-NAME-04 first, because of the security impact. Then 02, 05, 01, 03, with 07 last.

## 4. DOC-DEMO-01 and 02: README demos

**Input:**
- tape files in `docs/tapes/`: `scan.tape` and `safe-read.tape`
- canary data from `bench/canary_repo.py` with seed 0, never real secrets

**Output:**
- GIFs in `docs/assets/`, 20 s or less and 5 MB or less each
- alt text, and a text transcript below each GIF in the README

**CI:** a `demos` job runs `charmbracelet/vhs-action`. It is pinned to a release, and that pin is checked when this ships. The job fails if a GIF exceeds the limits, and runs on pushes that touch `docs/tapes/`.

**Scenes:**
1. `harpo scan` on the canary repo catches a password that regex-only scanners miss. The comparison uses `bench/compare_scanners.py` output, recorded once.
2. An agent-style read: `python -m Harpocrates.hooks` refuses the file, then `safe_read` returns placeholders. VHS records only a terminal, so this scene is scripted, not a live agent session. The README says so.

**Depends on** FR-NAME-02, because the tapes type `harpo`, so this goes after the rename.

**Test plan:**
- `test_doc_demo_01_tapes_use_only_canaries`: tapes reference only the canary repo or generated paths.
- The CI job is the acceptance test.
