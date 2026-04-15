# Harpocrates × TaskMaster — Session Handoff

**Last updated:** 2026-04-14
**Project root:** `/Users/joshuado/harpocrates`
**Purpose:** Brief a fresh session so it can pick up exactly where we left off.

---

## 1. Project context (30-second brief)

Harpocrates is a CLI-first, ML-powered secrets detector (Python). Main code at `Harpocrates/`:

| Path | What lives there |
|---|---|
| `Harpocrates/cli/__init__.py` | 702-line Typer CLI. Entry: `harpocrates = "Harpocrates.cli:main"` (wired in `pyproject.toml`). |
| `Harpocrates/core/` | `scanner.py`, `detector.py`, `result.py` |
| `Harpocrates/detectors/` | `regex_patterns.py`, `entropy_detector.py`, `filters.py` |
| `Harpocrates/ml/` | Two-stage XGBoost + LightGBM ensemble (`ensemble.py`, `features.py`, `verifier.py`) |
| `Harpocrates/api/` | FastAPI service (routers, middleware) — `services/` is empty |
| `Harpocrates/llm/` | LLM-based plausibility verifier |
| `Harpocrates/training/` | Dataset generators, two-stage training |
| `benchmark/`, `tests/`, `scripts/`, `docs/` | Harness, suites, automation, competitive analysis |

**Key prior docs (pre-TaskMaster):**
- `docs/roadmap.md` — long-range roadmap + gap analysis vs. TruffleHog/Gitleaks
- `docs/IMPLEMENTATION_PLAN.md` — detector expansion plan (target 160+ patterns)
- `docs/comparison.md` — competitor benchmark notes

**Pre-existing ML headline numbers:** ~88% precision, ~97% recall (two-stage ensemble).

---

## 2. What was built across recent sessions

### 2.1 Twelve PRDs under `.taskmaster/docs/`
Each PRD is self-contained (problem, goals, non-goals, current state with file-level references, proposed solution, acceptance criteria, task breakdown, risks, metrics).

| # | File | Scope |
|---|---|---|
| 01 | `prd-01-cli-restoration.md` | CLI hardening: `--show-secrets`, `--fail-on`, `verify` subcommand, redaction helper, README parity, smoke test |
| 02 | `prd-02-sarif-output.md` | SARIF 2.1.0 emitter + `--sarif` flag |
| 03 | `prd-03-github-action-precommit.md` | Composite GitHub Action + pre-commit hook |
| 04 | `prd-04-detector-expansion.md` | Expand detector count to 50+ (AI/ML, DevOps, SaaS, observability) |
| 05 | `prd-05-baseline-allowlist.md` | `.harpocrates-baseline`, inline ignore comments, config allowlist |
| 06 | `prd-06-git-history-scanning.md` | `--git-history`, commit-range scanning, blob-level dedupe |
| 07 | `prd-07-api-service-layer.md` | Extract `api/services/`, wire auth + rate limit, structured logs |
| 08 | `prd-08-active-verification.md` | Live verification for AWS, GitHub, Stripe, OpenAI, Slack |
| 09 | `prd-09-ml-pipeline-v2.md` | Contrastive/adversarial training data + retrained model v2 |
| 10 | `prd-10-performance-parallel.md` | Multiprocessing scanner + content-hash cache |
| 11 | `prd-11-release-v0.2.0.md` | PyPI + GHCR publishing, SBOM, sigstore signing |
| 12 | `prd-12-ci-quality-gates.md` | `.github/workflows/ci.yml` with ruff/mypy/pytest/coverage/self-scan |

`.taskmaster/docs/README.md` is the PRD index (user-edited to prepend priority guidance).

### 2.2 Priority order (agreed)
1. **PRD-01** — CLI (currently active in TaskMaster)
2. PRD-12 — CI gates (before detector/ML changes so regressions are caught)
3. PRD-02 — SARIF
4. PRD-05 — Baseline (needed before self-scan in CI blocks)
5. PRD-03 — GitHub Action
6. PRD-11 — Release v0.2.0
7. PRD-04 — Detector expansion
8. PRD-07 — API service layer
9. PRD-09 — ML v2
10. PRD-10 — Performance
11. PRD-06 — Git history scanning
12. PRD-08 — Active verification (last; highest safety risk)

### 2.3 TaskMaster install + config
- task-master-ai `0.43.1` installed globally via `npm install -g task-master-ai`.
- `.taskmaster/config.json` fixed: `modelId` must be `"sonnet"` (not `"claude-code/sonnet"`) for all three model slots. Using `provider: "claude-code"` so generation routes through the local Claude Code CLI — no API tokens billed.
- Auto-populated provider blocks present but dormant: `grokCli`, `ollamaBaseURL`, `bedrockBaseURL`, `claudeCode`, `codexCli`. Ignore unless you intentionally switch providers.

### 2.4 PRD-01 parsed into tasks
- `task-master parse-prd --input=.taskmaster/docs/prd-01-cli-restoration.md --num-tasks=10 --force`
- Output: `.taskmaster/tasks/tasks.json` with 10 tasks.
- Complexity analysis: `.taskmaster/reports/task-complexity-report.json` (threshold 5; scores 3–7).
- Expansion: `task-master expand --all` then `task-master expand --id=10 --num=7` (task 10 had to be retried after a rate-limit hit). **40 subtasks total** across tasks 1–7 and 10. Tasks 8 and 9 intentionally left flat (score 3).

---

## 3. Current task state

### Expanded (run `task-master show <id>` for subtask detail)
| Task | Title | Subtasks |
|---|---|---|
| 1 | Add `--show-secrets` flag | 4 |
| 2 | `--fail-on` severity gate | 5 |
| 3 | `verify` subcommand (LLM) | 6 |
| 4 | Reusable redaction helper | 5 |
| 5 | Rich table redaction fix | 3 |
| 6 | Lazy ML imports | 4 |
| 7 | CLI test coverage expansion | 6 |
| 10 | Smoke test install flow | 7 |

### Flat (left atomic by design)
- **Task 8:** Update README CLI docs (score 3)
- **Task 9:** Align `--ml-threshold` default to 0.19 (score 3)

### First recommended task
`task-master next` → **Task 1** (priority high, no dependencies).

---

## 4. How to resume

### Quick-start commands
```bash
cd /Users/joshuado/harpocrates

# See where we are
task-master list --with-subtasks
task-master next

# Start working on Task 1
task-master show 1
task-master set-status --id=1 --status=in-progress

# When a subtask is done
task-master set-status --id=1.1 --status=done

# When a parent is done
task-master set-status --id=1 --status=done

# Generate tasks for the next PRD (PRD-12 per priority order)
task-master parse-prd --input=.taskmaster/docs/prd-12-ci-quality-gates.md --num-tasks=10 --append --tag=ci
```

### Important files to re-read first
1. `.taskmaster/README.md` — this file
2. `.taskmaster/docs/README.md` — PRD index + priority guidance
3. `.taskmaster/docs/prd-01-cli-restoration.md` — active PRD
4. `.taskmaster/tasks/tasks.json` — source of truth for task state
5. `.taskmaster/reports/task-complexity-report.json` — scores + expansion prompts
6. `Harpocrates/cli/__init__.py` — where most PRD-01 work lands

---

## 5. Known quirks / gotchas

- **Rate limits:** Claude Code CLI calls share a quota with normal chat. When expanding all 10 tasks we hit the limit mid-run (reset message: "You've hit your limit · resets 8pm (America/Los_Angeles)"). task-master silently returns "completed" when this happens — the failure shows up as empty `subtasks: []`. Always verify after bulk ops with:
  ```bash
  python3 -c "import json; d=json.load(open('.taskmaster/tasks/tasks.json')); [print(t['id'], len(t.get('subtasks', []))) for t in d['master']['tasks']]"
  ```
- **Codebase analysis on by default:** `enableCodebaseAnalysis: true` in `.taskmaster/config.json` means every AI call uploads ~350k tokens of repo context. Useful, but slow and token-heavy. Disable for cheap ops.
- **CLI reality vs. PRD framing:** PRD-01 was originally framed as "CLI restoration from scratch" — but `Harpocrates/cli/__init__.py` already has a 702-line Typer app. The generated tasks correctly scope to *hardening* the existing CLI (redaction defaults, flags, tests), not building from zero.
- **`.taskmaster/reports/` directory** must exist before running `analyze-complexity`. task-master 0.43.1 does not auto-create it.
- **Grok/Ollama/Bedrock blocks** in `config.json` are dormant defaults. Don't delete unless you're tidying.

---

## 6. Session handoff on usage exhaustion (rule)

**When plan usage is about to be exhausted, the assistant MUST write a session-handoff log before the session terminates.** This preserves context for the next session and avoids re-deriving state from scratch.

### Trigger

- Usage gauge shows the current plan is near its limit (e.g., `/usage` / `/extra-usage` indicates imminent exhaustion), OR
- The assistant observes it may be interrupted mid-task (rate limit, quota, tool timeouts).

### Procedure

1. **Do not start new multi-step work.** Finish the smallest reversible unit in flight (single commit, single tool call).
2. **Use the extra-usage session** (spawned via `/extra-usage` login) to create the handoff log — this keeps the handoff itself from consuming the primary plan's remaining budget.
3. Write the log to `.taskmaster/sessions/` using one of:
   - **Markdown:** `session-YYYY-MM-DD-HHMM.md` (preferred when narrative context matters)
   - **JSON:** `session-YYYY-MM-DD-HHMM.json` (preferred when the next session is another agent that will machine-parse state)
4. Commit the log on its own branch (`docs/session-handoff-YYYY-MM-DD`) and push, so the next session can clone it down without needing local filesystem access.

### Required contents

At minimum the log MUST include:

| Field | Description |
|---|---|
| `ended_at` | ISO-8601 timestamp when the session stopped |
| `branch` | Active git branch at stop time |
| `last_commit` | SHA + title of the last commit made this session |
| `working_tree` | Output of `git status --short` (modified / untracked files) |
| `active_task` | TaskMaster task id + title (e.g., `1 — Add --show-secrets flag`) |
| `active_subtask` | TaskMaster subtask id + title, if applicable |
| `last_completed` | Last subtask or unit of work that finished cleanly |
| `next_step` | Single-sentence description of the next action for the new session |
| `open_questions` | Anything that was waiting on the user when the session stopped |
| `files_touched` | List of files modified this session, with line ranges if relevant |
| `tests_status` | Last test-suite outcome (`passed 46/46` or equivalent) |

### Markdown template

```markdown
# Session handoff — <YYYY-MM-DD HHMM>

- **Ended at:** 2026-04-14T21:30:00-07:00
- **Branch:** task/1-show-secrets-flag
- **Last commit:** 40d3b4c feat(cli): add --show-secrets flag for explicit token disclosure
- **Active task:** 1 — Add --show-secrets flag (PRD-01)
- **Active subtask:** 1.4 — Write comprehensive tests for all output combinations
- **Last completed:** 1.3 — Update JSON output to conditionally include token field
- **Tests:** 46/46 tracked core tests passing

## Working tree

```
 M Harpocrates/cli/__init__.py
 ?? .taskmaster/sessions/session-2026-04-14-2130.md
```

## Next step

Run `task-master set-status --id=1 --status=done` once the PR for `task/1-show-secrets-flag` is merged, then start Task 2 (`--fail-on` severity gate) off main.

## Open questions

- Should Task 2 land as one PR or be split by severity level?
```

### JSON template

```json
{
  "ended_at": "2026-04-14T21:30:00-07:00",
  "branch": "task/1-show-secrets-flag",
  "last_commit": {
    "sha": "40d3b4c",
    "title": "feat(cli): add --show-secrets flag for explicit token disclosure"
  },
  "active_task": { "id": 1, "title": "Add --show-secrets flag" },
  "active_subtask": { "id": "1.4", "title": "Write comprehensive tests" },
  "last_completed": "1.3 — Update JSON output to conditionally include token field",
  "next_step": "Open PR for task/1 branch, then start Task 2 off main.",
  "working_tree": [" M Harpocrates/cli/__init__.py"],
  "files_touched": [
    "Harpocrates/cli/__init__.py:27-200",
    "Harpocrates/core/result.py:76-90",
    "tests/test_cli.py:124-224"
  ],
  "tests_status": "passed 46/46 core",
  "open_questions": []
}
```

### On resume

The next session MUST start by reading the most recent `session-*` file in `.taskmaster/sessions/` before taking any other action.

---

## 7. Open follow-ups

- [ ] Start PRD-01 Task 1 (`--show-secrets` flag).
- [ ] When PRD-01 tasks are green, generate tasks for PRD-12 (`--append --tag=ci`).
- [ ] Decide whether to dogfood `harpocrates scan .` as a PRD-12 CI gate (depends on PRD-05 baseline).
- [ ] Consider removing the dormant provider blocks in `.taskmaster/config.json` for tidiness.

---

## 8. Directory map (TaskMaster workspace)

```
.taskmaster/
├── README.md                          ← this file
├── config.json                        ← modelId="sonnet", provider="claude-code"
├── docs/
│   ├── README.md                      ← PRD index + priority list
│   ├── prd-01-cli-restoration.md      ← active
│   ├── prd-02-sarif-output.md
│   ├── prd-03-github-action-precommit.md
│   ├── prd-04-detector-expansion.md
│   ├── prd-05-baseline-allowlist.md
│   ├── prd-06-git-history-scanning.md
│   ├── prd-07-api-service-layer.md
│   ├── prd-08-active-verification.md
│   ├── prd-09-ml-pipeline-v2.md
│   ├── prd-10-performance-parallel.md
│   ├── prd-11-release-v0.2.0.md
│   └── prd-12-ci-quality-gates.md
├── reports/
│   └── task-complexity-report.json
├── sessions/                          ← session-handoff logs (see §6)
│   └── session-YYYY-MM-DD-HHMM.md
└── tasks/
    └── tasks.json                     ← 10 parent tasks, 40 subtasks
```

---

## 9. Mandatory tool flow (skills + subagents)

**Background:** In the 2026-04-14 session the assistant pushed five PRD-01
branches without invoking any everything-claude-code (ECC) skill or reviewer
subagent, then re-reviewed them retroactively after the user objected
twice. This section exists so that mistake does not repeat.

### 9.1 Discover the ECC inventory (run on every fresh session)

```bash
# Confirm the marketplace path (3 install locations are normal)
find /Users/joshuado -maxdepth 5 -type d -name "everything-claude-code"

# Enumerate what's available
ls /Users/joshuado/.claude/plugins/marketplaces/everything-claude-code/agents/
ls /Users/joshuado/.claude/plugins/marketplaces/everything-claude-code/commands/
ls /Users/joshuado/.claude/plugins/marketplaces/everything-claude-code/skills/

# Search by topic (e.g. anything matching "security")
find /Users/joshuado/.claude/plugins/marketplaces/everything-claude-code/{agents,commands,skills} \
     -maxdepth 2 -iname "*security*"
```

The canonical root is
`/Users/joshuado/.claude/plugins/marketplaces/everything-claude-code/`.
A project-local mirror lives at
`/Users/joshuado/harpocrates/.claude/everything-claude-code/`. The
`harpocrates-*` skills load from a separate source and surface via system
reminders — invoke them with the `Skill` tool.

### 9.2 Required flow per task

| Phase | Tool | Trigger |
|---|---|---|
| Before any edit | `Skill` | Pick the matching skill — `python-patterns`, `python-testing`, `security-review`, `verification-loop`, or a `harpocrates-*` skill if relevant. State the choice in plain text first. |
| During edit | (direct tools) | Edit / Write / Bash as needed. |
| After any edit | `Agent` | Invoke the matching reviewer — `python-reviewer`, `security-reviewer`, `silent-failure-hunter`, `pr-test-analyzer`, `code-reviewer`. Run reviewers in parallel when independent. |
| After fix | `Skill: verification-loop` | Confirm the reviewer's flagged issue is actually closed. Do not assume the patch worked because tests pass. |
| Before push | `Agent: code-reviewer` (or language-specific) | Final independent pass on the diff that's about to ship. |

### 9.3 Trigger map (which subagent / skill for which signal)

| Signal | Tool to invoke |
|---|---|
| Touching `Harpocrates/cli/**` | `Agent: python-reviewer`, `Skill: python-patterns` |
| Touching `Harpocrates/utils/redaction.py` or any token rendering | `Agent: security-reviewer`, `Skill: security-review` |
| Touching `Harpocrates/ml/**` or `Harpocrates/training/**` | `Agent: python-reviewer`, `Agent: silent-failure-hunter` (ML failure modes silently degrade) |
| Touching `Harpocrates/api/**` | `Agent: security-reviewer` (auth, input validation), `Agent: python-reviewer` |
| Adding/changing tests | `Skill: python-testing`, `Agent: pr-test-analyzer`, `Skill: tdd-workflow` |
| README / docs change documenting unmerged features | `Agent: code-reviewer` (cross-check claims against actual code on `main`) |
| Build or import-graph changes | `Agent: python-reviewer`, run subprocess-based regression tests |
| Multiple branches need review | Launch reviewers **in parallel** in a single message |

### 9.4 Declare-before-acting

Before each task, output one line:

```
SKILL=<name or "none">  REVIEWER=<agent or "none">  REASON=<one phrase>
```

`none` is acceptable but must be justified. If neither is named, stop and
re-read this section.

### 9.5 Anti-patterns observed in 2026-04-14 session

- Spawning 5 reviewer subagents at the start, then doing all 4 fix patches
  with only the `Edit` tool — no follow-up reviewer pass to confirm fixes
  closed the flagged issues.
- Never invoking the `Skill` tool despite `harpocrates-security`,
  `python-patterns`, `python-testing`, `security-review`, and
  `verification-loop` being available.
- Treating the system reminders that surface `Skill` schemas as
  acknowledgments instead of as a prompt to use them.

### 9.6 Highest-leverage enforcement (recommended)

A `PreToolUse` hook on `Edit|Write` matchers that fails non-zero unless a
`Skill` tool call appears in the recent transcript. Hooks are the only
mechanism the assistant cannot rationalize past. Prompts and CLAUDE.md
guidance drift; hooks do not. See `~/.claude/rules/python/hooks.md` and
`~/.claude/rules/web/hooks.md` for shape.
