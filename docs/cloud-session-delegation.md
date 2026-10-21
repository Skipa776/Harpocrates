# Cloud session: delegating to Haiku agents

Read this first in the next cloud session. It sets how the main agent splits work between itself and Claude Haiku 5.5 sub-agents, so that the work stays cheap and each Haiku agent stays in the lower price band.

## Why

Claude Haiku 5.5 costs $0.10 / $0.50 per million tokens while a request's prompt is 100K tokens or fewer, and $0.50 / $2.50 above that (5×). Pricing is per request: the prompt counts the system prompt, tools and the agent's whole history so far. So a Haiku agent that keeps working in one context gets 5× more expensive once its context passes 100K. Its tokenizer also counts text as about 30% more tokens than Haiku 4.5's.

## Rules for the main agent

1. **Plan, split, delegate.** The main agent reads the request, writes the plan and breaks it into small tasks using the template below. It hands each task to a sub-agent spawned with `model: "haiku"`, unless the task is in the "Keep on the main agent" list.
2. **One task at a time per agent.** Send a Haiku agent one task. Read its report before sending the next task (by `SendMessage` to the same agent, or to a new one).
3. **Track every agent's tokens.** Keep a table: agent ID, task IDs done, tokens after each task. Prefer the token count the harness reports when an agent finishes. The agent's own report is the fallback.
4. **Hand off before the limit.** When an agent's context reaches **85K tokens**, or its next task would plausibly take it past 100K, don't send it another task. Ask it for a handoff note (format below), then spawn a fresh Haiku agent whose prompt is that note plus the next task. 85K leaves room for one more task's reading and output. Never let a Haiku agent continue past 100K.
5. **Check before accepting.** Run each task's done-check yourself, or read the diff, before marking the task done. A Haiku report is a claim, not proof.
6. **Run independent tasks in parallel** with separate agents. Never give two agents the same files.

## Keep on the main agent (do not delegate to Haiku)

- Specs, design decisions, and anything needing Josh's approval.
- Security-relevant logic: gate scanning, fail-closed paths, redaction, `hooks.py`, `allowlist.py`, `grants.py`, `harness.py`, `mcp/`. Haiku may write fixtures or docs for these, never the logic.
- The five parts Josh writes by hand (see `docs/design.md`, marked **(Josh writes)**). Nobody implements them.
- Reviews and commits. Per project rules: reviewer agents per commit, `ruff check .` before every commit, and an end-to-end test through the real entry point before a feature counts as done.

## Task template (every delegated task uses this)

```
TASK <id>: <one-line imperative title>
Requirement: <requirement ID, or "none">
Goal: <one or two sentences: what is true when this is done>
Files: <exact paths the agent may change; read-only paths listed separately>
Context: <the facts the agent needs, inline; do not say "see above">
Steps:
  1. <concrete step>
  2. <concrete step>
Done when: <a command that passes, e.g. python -m pytest -q cli/tests/test_x.py && ruff check .>
Out of scope: <what not to touch or change>
Report: use the report format in docs/cloud-session-delegation.md
```

A good task is small enough to finish in a few minutes, touches one to three files, and has a done-check that is a single command. If you can't write the done-check, the task isn't ready to delegate: split it or keep it.

## Report format (every Haiku agent ends every task with this)

```
TASK <id>: done | blocked | failed
Changed: <files>
Check: <the done-check command and its result, pass/fail with counts>
Notes: <anything the main agent must know, one or two lines>
Tokens: <current context size in tokens, as best known>
```

If blocked, say what is missing and stop. Don't guess, and don't widen the task's scope.

## Handoff note (when the main agent asks for one)

```
HANDOFF from <agent id> at <tokens> tokens
Done: <task IDs and one line each>
In progress: <task, how far, what is left>
State: <branch, uncommitted files, anything running>
Facts learned: <short list the next agent needs, e.g. a fixture's location, a quirk>
Next: <the next task ID>
```

The new agent's prompt is: this file's report format, the handoff note, then the next task in the template.

## Example

```
TASK G6-03: Add the FR-LOG-01 schema fixture
Requirement: FR-LOG-01
Goal: A JSONL fixture of five valid audit-log lines exists for the schema test to load.
Files: cli/tests/fixtures/audit_log_valid.jsonl (new). Read-only: docs/design.md (FR-LOG-01 row).
Context: Fields per line: time, layer, type, hash_prefix, source, action, session_id. Values are fakes; no secret-like strings.
Steps:
  1. Write five lines covering layers "read" and "gate" and actions "redact" and "block".
  2. Check every line parses with python -c "import json,sys;[json.loads(l) for l in open(sys.argv[1])]" <file>.
Done when: that command exits 0.
Out of scope: the test itself and any code under cli/Harpocrates/.
Report: use the report format in docs/cloud-session-delegation.md
```
