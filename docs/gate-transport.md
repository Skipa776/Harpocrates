# Gate transport notes (FR-GATE-09)

Oct 19, 2026 · status: **awaiting Josh's review. No gate code exists yet.** M2, step G1. For Claude Code and Codex CLI: which endpoints, which transport, which headers, what bypasses the base URL, and what the gate must do on the response path. Request bodies are in [gate-parser-spec.md](gate-parser-spec.md).

Every external claim cites a source from the table at the end, each checked 2026-10-19. A claim is **VERIFIED** (source named) or **TO VERIFY IN G2** (the mock-server capture that will confirm it). No agent was run against a recorder to write this; G2 does that, with fake credentials only.

## Summary

| | Claude Code | Codex CLI |
| --- | --- | --- |
| Request format | Anthropic Messages | OpenAI Responses |
| Route to the gate | `ANTHROPIC_BASE_URL` in the `env` block of `~/.claude/settings.json` | Custom `[model_providers.harpo]` in `~/.codex/config.toml` (recommended), or `openai_base_url` |
| Main endpoint | `POST /v1/messages?beta=true` | `POST {base_url}/responses` |
| Transport | HTTP request, SSE response (`stream: true`) | HTTP request, SSE response. WebSocket exists, but only for the built-in OpenAI provider |
| Request compression | None documented. TO VERIFY IN G2 | zstd, only for the built-in OpenAI provider with a ChatGPT login. Not for a custom provider |
| Streaming | Required event by event | Required |
| Upstream | `https://api.anthropic.com` | `https://api.openai.com/v1` (API key) or `https://chatgpt.com/backend-api/codex` (ChatGPT login) |

**Recommendation for v1.** Route Codex through a custom provider named anything but "OpenAI", with `supports_websockets` left off. That gives plain HTTP and SSE, no zstd, and full history in every request, so FR-GATE-04 holds without tracking server-side state (TO VERIFY IN G2: T4). The gate still decodes gzip and zstd (FR-INSTALL-04) and refuses WebSocket upgrades with a clear message, so the other routing path fails safe.

This corrects one line in `design.md` ("Request bodies are zstd-compressed by default; WebSocket mode exists"). It is true only for the built-in provider. See Codex below.

## Claude Code

| Fact | Status | Source |
| --- | --- | --- |
| Messages format: `POST /v1/messages?beta=true`. `POST /v1/messages/count_tokens` is optional; without it Claude Code estimates by characters | VERIFIED | C1 |
| Streaming is SSE, read event by event. A clean end before the final `message_delta` counts as a dropped connection. Buffering stalls it | VERIFIED | C1 |
| No WebSocket for inference | TO VERIFY IN G2 (capture shows no `Upgrade` request) | none documented |
| Request body compression | TO VERIFY IN G2 (capture of `Content-Encoding` on a long request) | none documented |
| `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS` can be set in the settings-file `env` block, and the settings-file value wins over a shell export | VERIFIED | C2 |
| Subscription login keeps working when only `ANTHROPIC_BASE_URL` is set. The OAuth capability travels in `anthropic-beta`; stripping it gives 401 | VERIFIED | C1, C4 |
| An `ANTHROPIC_BASE_URL` on a non-Anthropic host disables Remote Control | VERIFIED | C2 |
| A project `.claude/settings.json` `env` block applies only after the first-run wizard and trust prompt. Whether it can override a user-level base URL is open | Trust prompt VERIFIED. Precedence TO VERIFY IN G2 | C2 |
| First system block is an attribution block that the API strips by position | VERIFIED | C1 |

**Headers that must pass untouched** (C1, VERIFIED): `anthropic-version`, `anthropic-beta` (verbatim, never allowlisted per value, because the set changes with releases), `Authorization`, `x-api-key` (either or both, depending on the credential variable; `apiKeyHelper` sends both), and any `anthropic-*` header. Claude Code also sends `x-claude-code-session-id`, `x-claude-code-agent-id`, `x-claude-code-parent-agent-id`, and, if enabled, hint headers that carry "never prompt text or file contents" (C1). The gate may read the session id for the audit log (hashed). It forwards everything else.

**Side endpoints.**

| Request | Carries content? | Gate policy |
| --- | --- | --- |
| `POST /v1/messages` | Yes | Parse, scan, redact (spec) |
| `POST /v1/messages/count_tokens` | Yes: same `messages`, `system`, `tools` fields (A6, VERIFIED) | Same parser. On a block, Claude Code's reaction is TO VERIFY IN G2 (documented fallback covers only an absent endpoint) |
| `GET /v1/models?limit=1000` | No. Only when `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. A redirect counts as failure (C1, VERIFIED) | Pass. Never redirect |
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
| Feature differences from losing the OpenAI provider (model catalog, Codex-backend routes) | TO VERIFY IN G2 | None |

OpenAI ships a strict proxy using route A. It accepts exactly `POST /v1/responses`, injects the key, forwards all other headers, and answers 403 to everything else (CX3, VERIFIED). That is prior art for the route policy below.

Both settings are user-level only: a project `.codex/config.toml` ignores `model_provider`, `model_providers` and `openai_base_url` (X1, VERIFIED).

**Endpoints** (VERIFIED in source, CX4, CX1): relative to `base_url`, Codex calls `POST /responses` (inference; also the WebSocket path), `GET /models`, `POST /images/generations`, `POST /images/edits`, `POST /alpha/search`, `POST /memories/trace_summarize`, `POST /realtime/calls`, and a compaction call (path TO VERIFY IN G2). With a custom provider all of these go to the gate if Codex builds them from the provider base URL. TO VERIFY IN G2: which of them actually do.

| Request | Carries content? | Gate policy |
| --- | --- | --- |
| `POST /responses` | Yes | Parse, scan, redact (spec) |
| `POST /images/*`, `/alpha/search`, `/memories/trace_summarize`, compaction | Yes (prompts, queries, traces) | No parser in v1. **Block** with the route in the message. Add parsers if a canary session needs them |
| `POST /realtime/calls` | Audio and SDP, not scannable | **Block** |
| `GET /models` | No | Pass |
| WebSocket upgrade on `/responses` | Yes, as frames | **Refuse** with `426`, see below |
| Any other POST, PUT, PATCH or DELETE with a body | Unknown | **Block** |

**WebSocket.** The Responses WebSocket transport exists and is used by the built-in OpenAI provider (CX2, VERIFIED). Frames are JSON text messages `{"type":"response.create", ...}`. After the first turn Codex sends only new input items plus `previous_response_id`, and it may send a `generate: false` warm-up first (CX1, O2, VERIFIED). Codex has a session-scoped fallback to HTTP when WebSocket fails (`force_http_fallback`, CX1, VERIFIED). What triggers it, and whether a `426` on the upgrade triggers it, is TO VERIFY IN G2.

*Proposed v1 scope:* no WebSocket support. Route A avoids it entirely; route B degrades to HTTP if the fallback works. WebSocket support is a v1.1 decision (open question 1), because it means tracking `previous_response_id` chains and scanning per frame (spec, "previous_response_id").

**Compression.** Codex compresses with zstd only when `enable_request_compression` is on, the login uses the ChatGPT backend, and the provider is the built-in OpenAI one (CX1, VERIFIED). The WebSocket client lists a permessage-deflate dependency in source (CX5); whether it is negotiated is TO VERIFY IN G2. Route A avoids both. The gate decodes gzip and zstd anyway and blocks any other `Content-Encoding` (FR-INSTALL-04).

**Headers that must pass untouched** (VERIFIED in source unless noted): `Authorization: Bearer ...` (API key or ChatGPT access token), `ChatGPT-Account-ID` (ChatGPT login), `OpenAI-Organization` and `OpenAI-Project` (from `OPENAI_ORGANIZATION` and `OPENAI_PROJECT`, CX2), `OpenAI-Beta`, `version`, `x-codex-installation-id`, `x-codex-turn-state`, `x-codex-turn-metadata`, `x-codex-window-id`, `x-codex-parent-thread-id`, `x-openai-subagent`, `x-client-request-id` (CX1). `x-codex-turn-state` is returned by the server and sent back by Codex for sticky routing, so it must pass in both directions (semantics TO VERIFY IN G2). Which upstream a request needs follows the login: the gate picks `chatgpt.com/backend-api/codex` when `ChatGPT-Account-ID` is present, else `api.openai.com/v1` (CX2 shows the defaults by auth mode; the exact request shape under route A with a ChatGPT login is TO VERIFY IN G2 using a fake `auth.json`).

## Rules for all headers

| Rule | Why |
| --- | --- |
| Forward every request header unchanged except: hop-by-hop headers, `Host` (set to the upstream), `Content-Length` and `Content-Encoding` (fixed after a rewrite), and the gate token header | FR-INSTALL-05. The gateway guide says to treat `anthropic-*` as an open list (C1) |
| Strip the gate token (`X-Harpo-Token`) before forwarding | SEC-03. The token must never reach the provider |
| Never write a header value to the log or audit entry. The audit log holds no headers at all | FR-INSTALL-05, SEC-01 |
| Scan non-auth header values with the same scanner. A finding **blocks**; headers are opaque, so there is no safe rewrite | `x-codex-turn-metadata` may carry repository or path data. What it holds is TO VERIFY IN G2 (capture with a fake git remote such as `https://user:CANARY@example.test/r.git`) |
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

SSE is the shared transport: Messages events carry a named `event:` line plus matching JSON (A7). Responses events follow `response.*` names (O2). Codex's SSE client lives in `codex-api/src/sse/responses.rs` (CX4, VERIFIED to exist; wire details TO VERIFY IN G2).

## Traffic that bypasses the base URL (SEC-06)

None of this passes through the gate. [threat-model.md](threat-model.md) already lists it; this section adds sources and how `harpo doctor` notices a bypass.

| Agent | Not on the base URL | Source |
| --- | --- | --- |
| Claude Code | Metrics, error reports (Pro and Max sign-ins), feature flags, version checks, survey uploads, `/feedback` | C3, C2 (VERIFIED) |
| Claude Code | WebFetch domain safety check (hostname to `api.anthropic.com`), fast-mode availability check, sign-in (Console auth) | C3, C1 (VERIFIED) |
| Codex | Token refresh at `https://auth.openai.com/oauth/token` | CX6 (VERIFIED) |
| Codex | Analytics, optional OpenTelemetry exporters, ChatGPT login flow (`chatgpt_base_url`) | X1 (VERIFIED) |

Switches: `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` and `skipWebFetchPreflight` (C2, C3); Codex `analytics.enabled = false` and no `otel.exporter` (X1).

**How `harpo doctor` detects bypass** (FR-INSTALL-02, FR-PROOF-05). It never reads tokens or `auth.json` contents.

1. **Config lint.** Resolve the effective route per agent. Claude Code: managed, user, project and local settings `env` blocks plus the shell. Fail if any scope sets the base URL to anything but the gate, or sets `allowedProviders` away from it (C4, VERIFIED). Codex: effective `model_provider`, profile and `openai_base_url` in user config (project config cannot set them, X1). Residual: per-run `-c model_provider=...` flags and aliases are invisible to doctor.
2. **Canary round trip.** Run one non-interactive prompt per agent (`claude -p`, `codex exec`) that contains a generated canary. Pass only if the gate's audit log shows a request in the window with a redaction of the canary's type. A run that produced a reply with no gate entry is a bypass.
3. **Passive cross-check.** Compare the agent's own session records with the gate's audit log, using the same hashed session ids. Claude Code keeps transcripts under `~/.claude/projects/` (C3, VERIFIED) and sends `x-claude-code-session-id` (C1, VERIFIED). A session with assistant turns after the gate came up and no gate entry is a bypass. For Codex, the rollout directory and the request header that carries the session id are TO VERIFY IN G2.
4. **Optional, off by default.** List the agent process's open connections and flag provider hosts that are not the gate.

Doctor reports bypass as a failure with the agent name and the check that failed. Claude Code and Codex that are not running are skipped, not passed.

## Open questions for Josh

1. **WebSocket scope.** Proposed: not in v1; route A avoids it; route B refuses the upgrade and relies on Codex's HTTP fallback. If ChatGPT-login users need route B, WebSocket moves into M2. Accept?
2. **Gate token.** SEC-03 says per session. Claude Code's `env` block and Codex's `http_headers` are static files, and IDE-launched agents do not see shell variables (FR-INSTALL-06). Options: rewrite both configs on every `harpo guard up` (proposed), or a stable per-install token stored `0600`. The token stops other users and browser pages, not other processes of the same user, who can read the same files.
3. **Header scanning.** Block when a non-auth header value holds a secret (proposed), or only log? Depends on what `x-codex-turn-metadata` carries.
4. **Unknown POST routes block.** This will break Codex features that call other endpoints (image generation, memories) under route A until they get parsers. Acceptable for v1?
5. **Update `design.md`** harness table for Codex: zstd and WebSocket apply to the built-in provider only. Want me to edit it?

## To verify in G2

| # | Claim | Confirmed by |
| --- | --- | --- |
| T1 | Claude Code sends no WebSocket upgrade and no request `Content-Encoding` | Mock capture: scripted `claude -p` with a fake key, headers only |
| T2 | Project settings `env` base URL precedence against user settings | Mock capture with conflicting settings files |
| T3 | Claude Code's reaction to a gate 400 on `/v1/messages` and on `count_tokens` | Mock returns the planned block body |
| T4 | Codex under route A: which endpoints it calls on the gate, and whether any feature is lost | Mock capture: scripted `codex exec` with fake `OPENAI_API_KEY` |
| T5 | Route A with a ChatGPT login: path, `ChatGPT-Account-ID`, upstream choice | Mock capture with a fake `auth.json` in a temp `CODEX_HOME` |
| T6 | Codex under route B: WebSocket upgrade attempt, effect of `426`, fallback to HTTP | Mock refuses the upgrade; capture next request |
| T7 | zstd presence under route B and `permessage-deflate` negotiation | Mock capture of `Content-Encoding` and the WebSocket extension header |
| T8 | Contents of `x-codex-turn-metadata` | Mock capture with a fake git remote |
| T9 | `x-codex-turn-state` round trip, compaction endpoint path, session-id header and rollout path for doctor | Mock capture across two turns |
| T10 | Request bodies arrive chunked or with `Content-Length` | Mock capture (same as V7 in the parser spec) |

## Sources (all checked 2026-10-19)

| ID | URL |
| --- | --- |
| A6 | https://platform.claude.com/docs/en/api/messages/count_tokens |
| A7 | https://platform.claude.com/docs/en/build-with-claude/streaming |
| C1 | https://code.claude.com/docs/en/llm-gateway-protocol |
| C2 | https://code.claude.com/docs/en/llm-gateway-connect |
| C3 | https://code.claude.com/docs/en/data-usage |
| C4 | https://code.claude.com/docs/en/llm-gateway |
| O2 | https://developers.openai.com/api/docs/guides/websocket-mode.md |
| X1 | https://learn.chatgpt.com/docs/config-file/config-reference |
| X2 | https://learn.chatgpt.com/docs/auth |
| CX1 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/core/src/client.rs |
| CX2 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/model-provider-info/src/lib.rs |
| CX3 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/responses-api-proxy/README.md |
| CX4 | https://github.com/openai/codex/tree/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/codex-api/src/endpoint (and `.../codex-api/src/sse/responses.rs`) |
| CX5 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/codex-api/src/endpoint/responses_websocket.rs |
| CX6 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/login/src/auth/manager.rs |
