# Gate transport notes (FR-GATE-09)

Oct 19, 2026 · status: **decisions approved; G2 capture results added ("To verify in G2" at the end). No gate code exists yet.** M2, step G1. For Claude Code and Codex CLI: which endpoints, which transport, which headers, what bypasses the base URL, and what the gate must do on the response path. Request bodies are in [gate-parser-spec.md](gate-parser-spec.md).

Every external claim cites a source from the table at the end, each checked 2026-10-19. A claim is **VERIFIED** (source named, or a G2 probe run) or **TO VERIFY IN G2** (the mock-server capture that will confirm it). The first version was written without running an agent; G2 then ran Claude Code 2.1.303 and Codex CLI 0.163.0 against the mock with fake credentials only, in probe run `20261019T175710Z-g2` (`bench/results/proof/g2-probes-20261019T175710Z-g2.json`). Where a row below says "VERIFIED in G2", that run is the source. Open items are marked STILL OPEN with the reason.

## Summary

| | Claude Code | Codex CLI |
| --- | --- | --- |
| Request format | Anthropic Messages | OpenAI Responses |
| Route to the gate | `ANTHROPIC_BASE_URL` in the `env` block of `~/.claude/settings.json` | Custom `[model_providers.harpo]` in `~/.codex/config.toml` (recommended), or `openai_base_url` |
| Main endpoint | `POST /v1/messages?beta=true` | `POST {base_url}/responses` |
| Transport | HTTP request, SSE response (`stream: true`) | HTTP request, SSE response. WebSocket exists, but only for the built-in OpenAI provider |
| Request compression | None. VERIFIED in G2 (T1, V3) | zstd, only for the built-in OpenAI provider with a ChatGPT login. Not for a custom provider. VERIFIED in G2 (T7) |
| Streaming | Required event by event | Required |
| Upstream | `https://api.anthropic.com` | `https://api.openai.com/v1` (API key) or `https://chatgpt.com/backend-api/codex` (ChatGPT login) |

**Recommendation for v1.** Route Codex through a custom provider named anything but "OpenAI", with `supports_websockets` left off. That gives plain HTTP and SSE, no zstd, and full history in every request, so FR-GATE-04 holds without tracking server-side state (VERIFIED in G2: T4, V5). The gate still decodes gzip and zstd (FR-INSTALL-04) and refuses WebSocket upgrades with a clear message, so the other routing path fails safe.

This corrects one line in `design.md` ("Request bodies are zstd-compressed by default; WebSocket mode exists"). It is true only for the built-in provider. See Codex below.

## Claude Code

| Fact | Status | Source |
| --- | --- | --- |
| Messages format: `POST /v1/messages?beta=true`. `POST /v1/messages/count_tokens` is optional; without it Claude Code estimates by characters | VERIFIED | C1 |
| Streaming is SSE, read event by event. A clean end before the final `message_delta` counts as a dropped connection. Buffering stalls it | VERIFIED | C1 |
| No WebSocket for inference | VERIFIED in G2 (T1): no `Upgrade` header in any of the 21 requests Claude Code made in the G2 runs, and `POST /v1/messages` was the only endpoint it called | none documented |
| Request body compression | VERIFIED in G2 (T1, V3): no `Content-Encoding` on any request, including a 764,520-byte one; `Accept-Encoding: gzip, deflate, br, zstd` is sent for responses | none documented |
| `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS` can be set in the settings-file `env` block, and the settings-file value wins over a shell export | VERIFIED | C2 |
| Subscription login keeps working when only `ANTHROPIC_BASE_URL` is set. The OAuth capability travels in `anthropic-beta`; stripping it gives 401 | VERIFIED | C1, C4 |
| An `ANTHROPIC_BASE_URL` on a non-Anthropic host disables Remote Control | VERIFIED | C2 |
| A project `.claude/settings.json` `env` block applies only after the first-run wizard and trust prompt. Whether it can override a user-level base URL is open | Trust prompt VERIFIED (C2). Precedence VERIFIED in G2 (T2): **the project `.claude/settings.json` value won over both the user settings value and an exported shell variable** in `claude -p`, where no trust prompt is shown | C2 |
| First system block is an attribution block that the API strips by position | VERIFIED | C1 |

**Headers that must pass untouched** (C1, VERIFIED): `anthropic-version`, `anthropic-beta` (verbatim, never allowlisted per value, because the set changes with releases), `Authorization`, `x-api-key` (either or both, depending on the credential variable; `apiKeyHelper` sends both), and any `anthropic-*` header. Claude Code also sends `x-claude-code-session-id`, `x-claude-code-agent-id`, `x-claude-code-parent-agent-id`, and, if enabled, hint headers that carry "never prompt text or file contents" (C1). The gate may read the session id for the audit log (hashed). It forwards everything else.

**Side endpoints.**

| Request | Carries content? | Gate policy |
| --- | --- | --- |
| `POST /v1/messages` | Yes | Parse, scan, redact (spec) |
| `POST /v1/messages/count_tokens` | Yes: same `messages`, `system`, `tools` fields (A6, VERIFIED) | Same parser. On a block, Claude Code's reaction is STILL OPEN: no Claude Code session in G2 (print mode, 21 model requests, one with 764 KB) ever called it, so it could not be answered with a 400. The documented fallback covers only an absent endpoint |
| `GET /v1/models?limit=1000` | No. Only when `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. A redirect counts as failure (C1, VERIFIED). Not called in G2 (the variable was unset) | Pass. Never redirect |
| `HEAD /api/hello` | No. Connection warm-up, safe to reject (C1, VERIFIED) | Pass |
| Any other POST, PUT, PATCH or DELETE with a body | Unknown | **Block** (fail closed) with the route in the message |

## Codex CLI

Source reading below is of `openai/codex` at commit `e4568a3` (CX1 to CX4). Codex changes often; the version table of FR-INSTALL-09 should record the Codex version each claim was checked against.

**Two ways to route, and they behave differently.**

| | A. Custom provider (recommended) | B. `openai_base_url` |
| --- | --- | --- |
| Config | `[model_providers.harpo]` with `base_url = "http://127.0.0.1:PORT/v1"`, `wire_api = "responses"`, `requires_openai_auth = true` | `openai_base_url = "http://127.0.0.1:PORT/v1"` |
| Provider is "OpenAI" (`is_openai()` is a name match) | No. Any other name | Yes |
| WebSocket | Off: `supports_websockets` defaults to false | On: the built-in provider sets it true. VERIFIED, CX2 |
| zstd request bodies | Off: needs `is_openai()` and a ChatGPT-backend login. VERIFIED, CX1 | On with a ChatGPT login unless `features.enable_request_compression = false` (X1, VERIFIED) |
| API key or ChatGPT login | `requires_openai_auth = true` means OpenAI auth (either), and "suits LLM proxy setups". VERIFIED, X2 | Same |
| Feature differences from losing the OpenAI provider (model catalog, Codex-backend routes) | VERIFIED in G2 (T4, T5): with an API key Codex calls only `POST /v1/responses` and prints "Model metadata for `gpt-5` not found. Defaulting to fallback metadata" (no catalog request is made); with a ChatGPT login it also calls `GET /v1/models` and, on the ChatGPT backend base URL, `GET /backend-api/wham/accounts/check` and `/backend-api/wham/settings/user` | None found. Codex first tries a WebSocket upgrade on `/v1/responses` (T6) |

OpenAI ships a strict proxy using route A. It accepts exactly `POST /v1/responses`, injects the key, forwards all other headers, and answers 403 to everything else (CX3, VERIFIED). That is prior art for the route policy below.

Both settings are user-level only: a project `.codex/config.toml` ignores `model_provider`, `model_providers` and `openai_base_url` (X1, VERIFIED).

**Endpoints** (VERIFIED in source, CX4, CX1): relative to `base_url`, Codex calls `POST /responses` (inference; also the WebSocket path), `GET /models`, `POST /images/generations`, `POST /images/edits`, `POST /alpha/search`, `POST /memories/trace_summarize`, `POST /realtime/calls`, and a compaction call (path: `POST /v1/responses`, VERIFIED in G2, T9). With a custom provider all of these go to the gate if Codex builds them from the provider base URL. VERIFIED in G2 (T4): in every API-key session (baseline, corpus, long prompt, blocked request, compaction) the only endpoint called was `POST /v1/responses` (the compaction call is the same path, see T9). The other endpoints in this list were not called in G2; the ChatGPT-login session also reached `/backend-api/...` paths, which Codex builds from `chatgpt_base_url`, not from the provider base URL (T5).

| Request | Carries content? | Gate policy |
| --- | --- | --- |
| `POST /responses` | Yes | Parse, scan, redact (spec) |
| `POST /images/*`, `/alpha/search`, `/memories/trace_summarize`, compaction | Yes (prompts, queries, traces) | No parser in v1. **Block** with the route in the message. Add parsers if a canary session needs them |
| `POST /realtime/calls` | Audio and SDP, not scannable | **Block** |
| `GET /models` | No | Pass |
| WebSocket upgrade on `/responses` | Yes, as frames | **Refuse** with `426`, see below |
| Any other POST, PUT, PATCH or DELETE with a body | Unknown | **Block** |

**WebSocket.** The Responses WebSocket transport exists and is used by the built-in OpenAI provider (CX2, VERIFIED). Frames are JSON text messages `{"type":"response.create", ...}`. After the first turn Codex sends only new input items plus `previous_response_id`, and it may send a `generate: false` warm-up first (CX1, O2, VERIFIED). Codex has a session-scoped fallback to HTTP when WebSocket fails (`force_http_fallback`, CX1, VERIFIED). A `426` on the upgrade does trigger it: VERIFIED in G2 (T6). Codex printed one error line on stderr ("failed to connect to websocket: HTTP error: 426 Upgrade Required"), then sent the same request as `POST /v1/responses` in the same turn, and the turn completed.

*Proposed v1 scope:* no WebSocket support. Route A avoids it entirely; route B degrades to HTTP if the fallback works. WebSocket support is a v1.1 decision (open question 1), because it means tracking `previous_response_id` chains and scanning per frame (spec, "previous_response_id").

**Compression.** Codex compresses with zstd only when `enable_request_compression` is on, the login uses the ChatGPT backend, and the provider is the built-in OpenAI one (CX1, VERIFIED). The WebSocket client lists a permessage-deflate dependency in source (CX5); VERIFIED in G2 (T7): the upgrade request offers `Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits`. Whether it would be negotiated is STILL OPEN, because the mock refused every upgrade with 426 (the gate does too, decision 3). Route A avoids both. The gate decodes gzip and zstd anyway and blocks any other `Content-Encoding` (FR-INSTALL-04).

**Headers that must pass untouched** (VERIFIED in source unless noted): `Authorization: Bearer ...` (API key or ChatGPT access token), `ChatGPT-Account-ID` (ChatGPT login), `OpenAI-Organization` and `OpenAI-Project` (from `OPENAI_ORGANIZATION` and `OPENAI_PROJECT`, CX2), `OpenAI-Beta`, `version`, `x-codex-installation-id`, `x-codex-turn-state`, `x-codex-turn-metadata`, `x-codex-window-id`, `x-codex-parent-thread-id`, `x-openai-subagent`, `x-client-request-id` (CX1). `x-codex-turn-state` is returned by the server and sent back by Codex for sticky routing, so it must pass in both directions (VERIFIED in G2, T9: a value the mock set on the first response came back on every later request of the turn, and not on the first). Codex 0.163.0 also sends `x-codex-beta-features`, `session-id`, `thread-id`, `originator` and `x-client-request-id` on route A (T4), and on route B `version`, and with a ChatGPT login `x-codex-routing-hint: model=<model>` (T6, T7). Which upstream a request needs follows the login: the gate picks `chatgpt.com/backend-api/codex` when `ChatGPT-Account-ID` is present, else `api.openai.com/v1` (CX2 shows the defaults by auth mode). VERIFIED in G2 (T5, fake `auth.json`): under route A a ChatGPT login still sends the model request to `POST {provider base_url}/responses` with `Authorization` and `ChatGPT-Account-ID` (lower-case on the wire), no `Content-Encoding`. The path does not tell the gate which upstream to use; only the presence of `ChatGPT-Account-ID` does.

## Rules for all headers

| Rule | Why |
| --- | --- |
| Forward every request header unchanged except: hop-by-hop headers, `Host` (set to the upstream), `Content-Length` and `Content-Encoding` (fixed after a rewrite), and the gate token header | FR-INSTALL-05. The gateway guide says to treat `anthropic-*` as an open list (C1) |
| Strip the gate token (`X-Harpo-Token`) before forwarding | SEC-03. The token must never reach the provider |
| Never write a header value to the log or audit entry. The audit log holds no headers at all | FR-INSTALL-05, SEC-01 |
| Scan non-auth header values with the same scanner. A finding **blocks**; headers are opaque, so there is no safe rewrite | `x-codex-turn-metadata` may carry repository or path data. VERIFIED in G2 (T8, fake remote `https://probeuser:<fake>@example.test/r.git`): it holds the absolute workspace path as a key, `latest_git_commit_hash` (40 hex characters), the git `origin` URL **with the userinfo already stripped** (the planted password did not appear in any header or body), `has_changes`, ids, `sandbox`, `sandbox_mode`, `request_kind` and the model name; the same JSON is also in the body as `client_metadata["x-codex-turn-metadata"]`. Expect the 40-hex hash on every request |
| Scan the URL path and query string | A GET with a secret in the query is still egress |
| Set the token in `ANTHROPIC_CUSTOM_HEADERS` (Claude Code) and `http_headers` or `env_http_headers` (Codex) | VERIFIED as supported settings, C2, X1. Open question 2 |

## Response path (FR-GATE-06)

| Requirement | Status |
| --- | --- |
| Forward each chunk as received, flushed immediately. No buffering, no re-chunking that delays bytes | VERIFIED need, C1 |
| Keep `ping` events and all other events. A long thinking pause may have only pings on the wire, and Claude Code aborts on an idle stream | VERIFIED, C1 |
| Pass status, body and headers unchanged, including error bodies. Claude Code recovers from some 400s by matching the upstream's wording | VERIFIED, C1 |
| Pass `retry-after`, `x-should-retry`, `anthropic-ratelimit-unified-*` and `content-type: text/event-stream` unchanged | VERIFIED, C1 |
| No gate-side idle timeout shorter than the client's. Codex has its own `stream_idle_timeout_ms` | VERIFIED setting exists, X1 |
| If the upstream drops mid-stream, abort the downstream connection. Do not invent a clean ending or synthetic events | Design choice; the client treats a clean end before `message_delta` as a drop and retries (C1) |
| If the agent disconnects, cancel the upstream request | Design choice |
| Response bodies are not scanned or edited in v1. The only read is a tee that records `response.id` values for the previous-response policy | FR-GATE-06 |
| Gate-originated blocks use the provider's error shape. They must not contain the recovery phrases listed in the spec | Spec, open question 8 |
| Event schemas may grow ("new event types may be added", A7). The gate never parses them for forwarding | VERIFIED, A7 |

SSE is the shared transport: Messages events carry a named `event:` line plus matching JSON (A7). Responses events follow `response.*` names (O2). Codex's SSE client lives in `codex-api/src/sse/responses.rs` (CX4, VERIFIED to exist). VERIFIED in G2: both agents completed every scripted turn against the mock's SSE streams, which follow the documented event sequences (Messages: `message_start`, `content_block_*`, `ping`, `message_delta`, `message_stop`; Responses: `response.created`, `response.in_progress`, `response.output_item.*`, content and reasoning-summary events, `response.completed`), sent as chunked SSE.

## Traffic that bypasses the base URL (SEC-06)

None of this passes through the gate. [threat-model.md](threat-model.md) already lists it; this section adds sources and how `harpo doctor` notices a bypass.

| Agent | Not on the base URL | Source |
| --- | --- | --- |
| Claude Code | Metrics, error reports (Pro and Max sign-ins), feature flags, version checks, survey uploads, `/feedback` | C3, C2 (VERIFIED) |
| Claude Code | WebFetch domain safety check (hostname to `api.anthropic.com`), fast-mode availability check, sign-in (Console auth) | C3, C1 (VERIFIED) |
| Codex | Token refresh at `https://auth.openai.com/oauth/token` | CX6 (VERIFIED) |
| Codex | Analytics, optional OpenTelemetry exporters, ChatGPT login flow (`chatgpt_base_url`) | X1 (VERIFIED) |
| Codex | Found in G2: a plugin-catalog sync that runs `git ls-remote https://github.com/openai/plugins.git HEAD` from a fresh `CODEX_HOME` with default features, plus plugin, MCP and analytics calls on `chatgpt_base_url` when logged in with ChatGPT. The harness switches these off in its throwaway config (`plugins`, `remote_plugin`, `plugin_sharing`, `apps`, `daemon_auto_start`, `in_app_updates`, `[analytics]`) | G2 (`bench/agent_runner.py`, `CODEX_QUIET`) |

Switches: `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` and `skipWebFetchPreflight` (C2, C3); Codex `analytics.enabled = false` and no `otel.exporter` (X1).

**How `harpo doctor` detects bypass** (FR-INSTALL-02, FR-PROOF-05). It never reads tokens or `auth.json` contents.

1. **Config lint.** Resolve the effective route per agent. Claude Code: managed, user, project and local settings `env` blocks plus the shell. Fail if any scope sets the base URL to anything but the gate, or sets `allowedProviders` away from it (C4, VERIFIED). Codex: effective `model_provider`, profile and `openai_base_url` in user config (project config cannot set them, X1). Residual: per-run `-c model_provider=...` flags and aliases are invisible to doctor.
2. **Canary round trip.** Run one non-interactive prompt per agent (`claude -p`, `codex exec`) that contains a generated canary. Pass only if the gate's audit log shows a request in the window with a redaction of the canary's type. A run that produced a reply with no gate entry is a bypass.
3. **Passive cross-check.** Compare the agent's own session records with the gate's audit log, using the same hashed session ids. Claude Code keeps transcripts under `~/.claude/projects/` (C3, VERIFIED) and sends `x-claude-code-session-id` (C1, VERIFIED). A session with assistant turns after the gate came up and no gate entry is a bypass. For Codex: VERIFIED in G2 (T9) that the rollout is `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread-id>.jsonl` and that the `thread-id` request header equals the thread id in that file name; the `session-id` header is also sent (its value is not readable in the recording because the mock stores credential-named headers as `<present>`). For Claude Code: the session id in the request body (`metadata.user_id`, a JSON string) equals the transcript file name `~/.claude/projects/<project-slug>/<session-id>.jsonl` (VERIFIED in G2); `x-claude-code-session-id` is sent too.
4. **Optional, off by default.** List the agent process's open connections and flag provider hosts that are not the gate.

Doctor reports bypass as a failure with the agent name and the check that failed. Claude Code and Codex that are not running are skipped, not passed.

## SEC-10: a repo cannot move the agent off the gate

Oct 19, 2026. The hook (`hooks.py`) runs the check in `cli/Harpocrates/gatecheck.py` first on every Claude Code tool call. The code and its module docstring are the one definition; this section is the summary.

**Install record.** `~/.harpo/install.json` gets one more top-level entry, written by `harpo guard up` (G7), read by the hook:

```json
"gate": {"enabled": true, "port": 47321, "token_header": "x-harpo-gate-token"}
```

| Case | Hook |
| --- | --- |
| No record, no `gate` entry, `"enabled": false`, or `~/.harpo` is a symlink or a file (SEC-09) | No gate installed. Nothing changes |
| `"enabled": true` with an integer `port` of 1 to 65535 and a valid HTTP header name in `token_header` | The check runs |
| A `gate` entry that is not an object, `enabled` that is not true or false, a bad port or header; a record that cannot be read at all (permissions, a folder, over 1 MiB); a record that is not strict JSON but contains the quoted word `"gate"` (a damaged entry) | Every tool call is refused (fail closed), naming `~/.harpo/install.json`. Without the entry Harpo cannot tell whether the gate is on |
| A record that is not strict JSON and does not contain `"gate"`, or is JSON but not an object | No gate entry can be there, so no gate installed. Existing state-folder tests keep junk in `install.json` and expect the hook not to change |

**What is a bypass.** For each variable the effective value is the one from the highest-precedence source that sets it, an empty string included (a settings file cannot remove a variable, so an empty value is how one overrides it, C6).

| Variable | Bypass when |
| --- | --- |
| `ANTHROPIC_BASE_URL` | Not exactly `http://<localhost, 127.0.0.1 or [::1]>:<gate port>`: https, a path (even `/`), query, user info, a missing port or any other host fail. Also when no source sets it (Claude Code then calls the provider) |
| Any `CLAUDE_CODE_USE_<NAME>`: Bedrock, Vertex, Foundry, Mantle, Anthropic AWS and whatever a later release adds (matched by pattern, in any casing) | On. Claude Code documents `1`, `true`, `yes`, `on` as on and `0`, `false`, `no`, `off` as off, in any casing (C6); empty counts as unset for provider selection. The hook treats anything except empty and the four off words as on, the cautious reading of an unrecognised word. Two names in the C6 table are not providers and are skipped: `CLAUDE_CODE_USE_NATIVE_FILE_SEARCH` and `CLAUDE_CODE_USE_POWERSHELL_TOOL` (documented with `=1`) |
| `ANTHROPIC_CUSTOM_HEADERS` | No `Name: Value` line with the record's `token_header` and a non-empty value. Not set counts as lacking it |

**Precedence, highest first** (C5, checked 2026-10-19): managed settings (`managed-settings.json`, then `managed-settings.d/*.json` in name order, a later file winning, C7), `--settings`, `<repo>/.claude/settings.local.json`, `<repo>/.claude/settings.json`, `~/.claude/settings.json`; then the hook's own environment, because an `env` entry replaces the shell's value in the process environment (C6). G2 (T2) saw the same: the project value beat the user value and an exported variable. The hook reads the files above except `--settings`, MDM profiles, the Windows registry and server-managed settings; Claude Code copies what those set into the process environment, which the hook does read. The settings are looked up under the event's `cwd` and under `CLAUDE_PROJECT_DIR`, because a `cd` can move the `cwd` away from where Claude Code started.

**Refusal.** A `deny` with one plain line naming each setting file (and the line, when the key is written plainly) and the host and port of a wrong URL, never a path, query, user info, header or token. Every permission mode and `--ask` give the same deny: there is no `ask` code path, no desktop notice, and nothing reads further input. A settings file that exists but cannot be used (not a regular file, over 1 MiB, not strict JSON: duplicate keys, NaN, BOM, trailing text, a non-text value for one of the four variables) is itself the refusal, and then only such files are named, because the effective values are unknown. Files are opened without blocking, so a pipe planted in a repo cannot hang the hook (a hook that times out is read as an allow). Cost: about 0.1 ms with the gate on and two settings files, 0.015 ms with no gate (500 calls each, Python 3.12).

**Codex.** A project `.codex/config.toml` cannot set `model_provider`, `model_providers` or `openai_base_url` (X1; the Codex section above), so a repo cannot move Codex off the gate that way, and the check never opens a project `.codex` file (`test_sec_10_codex_project_config_is_never_read`). The hook skips the check for Codex events, which carry `turn_id` and `model` (X3), because Codex does not read Claude Code settings. An event with both fields is never judged; one without them is judged as Claude Code, which can only refuse more.

**`harpo doctor`** (FR-PROOF-05) does not exist yet (G7, FR-INSTALL-02). `gatecheck.bypass_findings(project_dir)` returns one finding per bypassing setting (or one for a bad record or file) and is what doctor will call and fail on.

**Not covered, for the next steps.**

- The hook only sees the tools its matcher names. `harness.claude_code_settings` registers `Read|Grep|Bash|Edit|MultiEdit|Write|NotebookEdit`, so a tool outside that list (`WebFetch`, `Task`, MCP tools) never reaches the hook. `decide` refuses any tool name it is given; widening the matcher is a one-line change in `harness.py`, not made here.
- `harpo install` rewrites the record from its own sections only (`install._run`), so it drops a `gate` entry written by `harpo guard up`. G7 must keep the entry when it rewrites the record, or the check goes quiet.
- `ANTHROPIC_AWS_BASE_URL` (C6) is an endpoint override beside `ANTHROPIC_BASE_URL`; it only matters with `CLAUDE_CODE_USE_ANTHROPIC_AWS` on, which is now a bypass by itself.
- Per-run flags and the process's own `HTTPS_PROXY` are invisible to the hook.

## Open questions for Josh (decided Oct 19, 2026: see Decisions in `docs/gate-parser-spec.md`)

1. **WebSocket scope.** Proposed: not in v1; route A avoids it; route B refuses the upgrade and relies on Codex's HTTP fallback. If ChatGPT-login users need route B, WebSocket moves into M2. Accept?
2. **Gate token.** SEC-03 says per session. Claude Code's `env` block and Codex's `http_headers` are static files, and IDE-launched agents do not see shell variables (FR-INSTALL-06). Options: rewrite both configs on every `harpo guard up` (proposed), or a stable per-install token stored `0600`. The token stops other users and browser pages, not other processes of the same user, who can read the same files.
3. **Header scanning.** Block when a non-auth header value holds a secret (proposed), or only log? Depends on what `x-codex-turn-metadata` carries.
4. **Unknown POST routes block.** This will break Codex features that call other endpoints (image generation, memories) under route A until they get parsers. Acceptable for v1?
5. **Update `design.md`** harness table for Codex: zstd and WebSocket apply to the built-in provider only. Want me to edit it?

## To verify in G2

Results, Oct 19, 2026. Probe run `20261019T175710Z-g2` (`bench/results/proof/g2-probes-20261019T175710Z-g2.json`); baseline run `20261019T175822Z-agent-read-hook-s0`. Claude Code 2.1.303 (API key path: `x-api-key` sent, no `Authorization`), Codex CLI 0.163.0, fake keys, mock on 127.0.0.1, throwaway homes. Header and field names and shapes only.

| # | Claim | Result |
| --- | --- | --- |
| T1 | Claude Code sends no WebSocket upgrade and no request `Content-Encoding` | **VERIFIED**. 21 model requests, no `Upgrade` header, no `Content-Encoding`, up to 764,520 bytes (probes `claude-corpus`, `claude-long`, baseline). Headers sent: `accept`, `accept-encoding`, `anthropic-beta`, `anthropic-dangerous-direct-browser-access`, `anthropic-version`, `connection`, `content-length`, `content-type`, `host`, `user-agent`, `x-api-key`, `x-app`, `x-claude-code-session-id`, `x-stainless-*` |
| T2 | Project settings `env` base URL precedence against user settings | **VERIFIED, and it matters.** With `ANTHROPIC_BASE_URL` set in user settings and in `<repo>/.claude/settings.json`, the request went to the project value. It also went to the project value when the variable was exported in the shell (probe `claude-settings`, print mode, no trust prompt). A repository can therefore move Claude Code off the gate. `harpo doctor` must lint project settings, and the threat model should say so. Interactive mode shows a trust prompt first (C2); not re-checked here |
| T3 | Claude Code's reaction to a gate 400 on `/v1/messages` and on `count_tokens` | **`/v1/messages`: VERIFIED**; **`count_tokens`: STILL OPEN** (never called, see above). Planned body: HTTP 400, `invalid_request_error`, the message below. Claude Code (probe `claude-block-400`): prints the message to the user (`result` event with `is_error: true`, `api_error_status: 400`, `terminal_reason: api_error`), exits 1, and **sends the request once more** with `thinking.display` removed and the `thinking-display-updates-2026-08-18` value removed from `anthropic-beta` (`x-stainless-retry-count` stays 0, so it is Claude Code's own fallback, not the SDK's). A block mid-session (after a tool result) was the same: 1 request answered, 2 blocked. So a gate block is seen twice and must block both. Codex (probe `codex-block-400`): one request, no retry, an `error` event with the message, `turn.failed`, exit 1. Test message used: "Harpo blocked this request before it left your machine (reason HARPO-BLOCK-UNKNOWN-TYPE). Run `harpo doctor` for details." (no recovery phrase) |
| T4 | Codex under route A: which endpoints it calls on the gate, and whether any feature is lost | **VERIFIED**. API key via `CODEX_API_KEY` (not `OPENAI_API_KEY`: with `requires_openai_auth = true` Codex 0.163.0 sent no `Authorization` when only `OPENAI_API_KEY` was set; an `env_key` provider setting also worked, in an ad-hoc run): only `POST /v1/responses`, `Content-Length`, no encoding, no WebSocket. No catalog request, so Codex prints "Model metadata for `gpt-5` not found. Defaulting to fallback metadata" and continues. In earlier ad-hoc runs with Codex's default features on (before the harness switched them off), a fresh `CODEX_HOME` also ran a GitHub plugin sync and, logged in with ChatGPT, made plugin, MCP and analytics calls (see the bypass table); those runs are not in the committed report. Headers: `accept`, `authorization`, `content-length`, `content-type`, `host`, `originator`, `session-id`, `thread-id`, `user-agent`, `x-client-request-id`, `x-codex-beta-features`, `x-codex-turn-metadata`, `x-codex-window-id`, plus `x-codex-turn-state` once the server sets it |
| T5 | Route A with a ChatGPT login: path, `ChatGPT-Account-ID`, upstream choice | **VERIFIED** (fake `auth.json` with unsigned tokens expiring in 2100, refresh URL and `chatgpt_base_url` on the mock). Model request: `POST /v1/responses` on the provider base URL, with `authorization` and `chatgpt-account-id`, no encoding. Also `GET /v1/models` (twice) and, under `chatgpt_base_url`, `GET /backend-api/wham/accounts/check` and `GET /backend-api/wham/settings/user` (answered 404 by the mock; the turn completed). The gate cannot choose the upstream from the path; `chatgpt-account-id` presence is the signal |
| T6 | Codex under route B: WebSocket upgrade attempt, effect of `426`, fallback to HTTP | **VERIFIED** (probes `codex-route-b-apikey`, `codex-route-b-chatgpt`). `openai_base_url` on the mock: the first request is `GET /v1/responses` with `Upgrade: websocket`; the mock answered 426; Codex logged "failed to connect to websocket: HTTP error: 426 Upgrade Required" and then sent `POST /v1/responses` in the same turn, which completed |
| T7 | zstd presence under route B and `permessage-deflate` negotiation | **zstd VERIFIED; negotiation STILL OPEN.** Route B with a fake ChatGPT login: the HTTP request has `Content-Encoding: zstd`, decoded by the mock with the `zstandard` module. Route B with an API key: no encoding. The upgrade request offered `Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits`; the mock never accepted an upgrade, so what would be negotiated is open (and moot while the gate refuses upgrades) |
| T8 | Contents of `x-codex-turn-metadata` | **VERIFIED** (probe `codex-corpus`, fake remote with a planted password). Shape: `installation_id`, `session_id`, `thread_id`, `turn_id`, `root_turn_id`, `window_id`, `window_number`, `context_window_id`, `agent_name`, `request_kind`, `history_initialization`, `thread_source`, `turn_trigger`, `turn_started_at_unix_ms`, `model`, `sandbox`, `sandbox_mode`, `analytics_enabled`, `auto_review_enabled`, `node_repl_auto_review_required`, `node_repl_disabled`, and `workspaces{<absolute path>: {has_changes, latest_git_commit_hash (40 hex), associated_remote_urls{origin}}}`. The origin URL had its userinfo stripped: the planted password was in no header and no body. The same JSON is in the body as `client_metadata["x-codex-turn-metadata"]` |
| T9 | `x-codex-turn-state` round trip, compaction endpoint path, session-id header and rollout path for doctor | **VERIFIED.** Turn state: sent back on requests 2 to 4 with the value the mock set on response 1, not on request 1. Compaction (probe `codex-compaction`, `model_auto_compact_token_limit = 2000`): Codex compacts with `POST /v1/responses` (the same path), marked `request_kind: "compaction"` in `x-codex-turn-metadata`, with `tools: []`; no other endpoint. Because the mock's usage figures never shrink, Codex compacted in a loop (288 compaction requests in 15 s) until the probe killed it. Session id: `thread-id` header equals the id in the rollout file name `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread-id>.jsonl`; `session-id` is also sent. Claude Code: the `session_id` inside `metadata.user_id` equals the transcript name `~/.claude/projects/<project-slug>/<session-id>.jsonl` |
| T10 | Request bodies arrive chunked or with `Content-Length` | **VERIFIED**, same as V7: `Content-Length` on all 615 model requests, none chunked, up to 1,039,744 bytes |
| T11 | What the agents' own image tools and MCP results put in a request (FR-GATE-07) | **VERIFIED** (suite `g5`, run `20261020-gate-e2e-g5g`; Claude Code 2.1.304, Codex 0.163.0). Details below the table. |

**T11 details.**
- **A real PNG.** Claude Code's `Read` sends it as an `image` block (base64 source) inside the `tool_result`. Codex's `view_image` sends it as an `input_image` data URL inside the `function_call_output`. Both resend it in every later request of the session.
- **A text file saved as `notes.png`.** Neither agent base64-encodes it; both check the bytes first.
  - Claude Code returns a text error that quotes the first 8 bytes in hex.
  - Codex returns `unable to process image`.

  Those 8 bytes reach the gate as ordinary text, which it scans.
- **MCP results.** These came from `bench/mcp_image_server.py`, which returns any file as an image or as a PDF resource without checking it.
  - Claude Code, given a non-image as an image, tries to resize it and then sends a text note. It saves a binary resource to disk and sends a text pointer. It never sent the bytes.
  - Codex sends a real MCP image as an `input_image` data URL, and replaces a non-image with `image content omitted because it could not be processed`.
  - Codex sends an embedded resource as **JSON text**: the `{"type":"resource",...,"blob":"<base64>"}` object goes inside an `input_text` part. So the one agent path that carries a file as base64 lands in a text field, where the gate's media rule does not apply. See the G5 spec's end-to-end result.


## Sources (all checked 2026-10-19)

| ID | URL |
| --- | --- |
| A6 | https://platform.claude.com/docs/en/api/messages/count_tokens |
| A7 | https://platform.claude.com/docs/en/build-with-claude/streaming |
| C1 | https://code.claude.com/docs/en/llm-gateway-protocol |
| C2 | https://code.claude.com/docs/en/llm-gateway-connect |
| C3 | https://code.claude.com/docs/en/data-usage |
| C4 | https://code.claude.com/docs/en/llm-gateway |
| C5 | https://code.claude.com/docs/en/settings ("Settings precedence") |
| C6 | https://code.claude.com/docs/en/env-vars ("Precedence"; the variable table) |
| C7 | https://code.claude.com/docs/en/managed-settings ("Where each mechanism stores the policy", "managed-settings.d") |
| O2 | https://developers.openai.com/api/docs/guides/websocket-mode.md |
| X1 | https://learn.chatgpt.com/docs/config-file/config-reference |
| X2 | https://learn.chatgpt.com/docs/auth |
| X3 | https://learn.chatgpt.com/docs/hooks (PreToolUse input; `model` and `turn_id` are Codex-specific) |
| CX1 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/core/src/client.rs |
| CX2 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/model-provider-info/src/lib.rs |
| CX3 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/responses-api-proxy/README.md |
| CX4 | https://github.com/openai/codex/tree/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/codex-api/src/endpoint (and `.../codex-api/src/sse/responses.rs`) |
| CX5 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/codex-api/src/endpoint/responses_websocket.rs |
| CX6 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/login/src/auth/manager.rs |
