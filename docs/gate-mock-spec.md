# G2 spec: mock provider and canary harness

Oct 19, 2026 · status: **awaiting Josh's approval** · requirements: Evaluation plan (test harness), FR-PROOF-04, and the TO VERIFY items in `docs/gate-parser-spec.md` (V1, V3–V7) and `docs/gate-transport.md` (T1–T10)

Built before the proxy (`M2_PLAN.md` step G2), so every later gate step is measured. Nothing here talks to a real provider.

## 1. Mock provider (`bench/mock_provider.py`)

**What it is.** One Python file using only the standard library (`http.server.ThreadingHTTPServer`). It binds `127.0.0.1` only, on a port chosen at start, and makes no outbound connection.

**Inputs:**

| Route | Behavior |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages. Returns a valid reply as JSON, or as an SSE stream when `stream: true` |
| `POST /v1/messages/count_tokens` | Anthropic token count. Returns a fixed count |
| `POST /v1/responses` | OpenAI Responses. Returns a valid reply as JSON, or as SSE when `stream: true` |
| `GET /v1/models` | A fixed model list |
| Anything else | Recorded, then answered `404` |

**Scripted replies.** An optional script file lists, turn by turn, what the mock answers. For example: turn 1 asks for a `Read` of `.env`, turn 2 a `Bash` `cat config/settings.py`, turn 3 says "done". This drives an agent through every read path without a real model. With no script, every reply is a short text turn.

**Output: the recording** is a JSONL file with one line per request:
- time, method and path
- every header except credentials, so T1 to T10 can be checked
- body size, SHA-256 of the raw body, and the decompressed body
- `Content-Encoding` and transfer encoding (chunked or not)

**Credentials are never written.** Any header whose name contains `auth`, `key`, `token`, `cookie`, `secret` or `session`, plus `chatgpt-account-id`, is stored only as `<present>`. A test proves no header value outside that allowlist could hold a credential. Bodies are stored, because the harness counts canaries in them. Canaries are fake, but a live session also sends the agent's own system prompt and memory files. So recordings go to `bench/results/mock/<run>/`, which is gitignored and never committed.

**Failure behavior.** A malformed or oversized body (over 64 MiB) is still recorded (size and hash only) and answered `400`. The mock never crashes on bad input. It shuts down cleanly on SIGINT.

## 2. Canary harness (`bench/canary_harness.py`)

**Inputs:** a seed, a mode (`direct` or `agent`), and a target (`mock`, or later `gate` → `mock`).

**Modes:**
- **`direct`:** builds synthetic requests in both formats, one per block and item type in the parser spec, each carrying a canary from `bench/canaries/secret_templates.py`. It sends them straight to the target. This is the corpus for the `test_fr_gate_02_*` parser tests.
- **`agent`:** generates the seeded canary repo (`bench/canary_repo.py`). Then it runs `claude -p` and `codex exec` non-interactively in that repo against the mock, using the mock's scripted tool calls.

**Isolation in `agent` mode:**
- each run gets a throwaway `HOME`
- a fake API key is set, so no real login or OAuth token can be used
- Harpo is installed into that throwaway home through a test-side bootstrap that points `paths.home()` at it, so your real `~/.harpo` is never touched (SEC-08)

The session-level real-home guard wraps every run.

**Output:** `bench/results/proof/<run>.json` lists:
- canaries planted and canaries received, by kind and by known-format vs ML-only
- which request and which block or item each received canary was found in
- the harness version and the agent versions

Canary values are matched in memory and never written to the report.

**Failure behavior.** The run exits non-zero when any known-format canary is received through a path the target claims to protect. Before the gate exists, that's the read-tool path. Every run publishes its counts, failing or not. If `claude` or `codex` is missing, `agent` mode reports that and skips. It never passes silently.

## 3. What G2 verifies

These verify items can't be settled from docs alone; the captures settle them:
- **Parser spec:** V1 (attribution prefix), V3 (no `Content-Encoding`), V4 (thinking blocks re-sent), V5 (no `previous_response_id` on route A), V6 (the real block corpus) and V7 (chunked vs `Content-Length`).
- **Transport notes:** T1 to T10.

Results are written back into the two docs as VERIFIED, with the run ID.

**V2 stays yours.** Whether providers accept rewritten text in blocks they produced needs a live API call with a fake secret, made by you.

## 4. Test plan, tests first

All tests are named after FR-PROOF-04, which this infrastructure serves:
- `test_fr_proof_04_mock_binds_loopback_only`
- `test_fr_proof_04_mock_answers_both_formats`: JSON and SSE replies parse with each provider's official response shape.
- `test_fr_proof_04_mock_never_records_credentials`: requests carrying fake `x-api-key`, `Authorization` and `cookie` headers leave only `<present>` in the recording.
- `test_fr_proof_04_mock_records_unknown_routes`
- `test_fr_proof_04_mock_survives_bad_bodies`: malformed JSON, oversized and wrong-encoding bodies.
- `test_fr_proof_04_harness_direct_counts_every_block_type`: against the bare mock, every planted canary is received, which proves the counter works.
- `test_fr_proof_04_harness_reports_without_values`: no canary value appears in the report.
- `test_fr_proof_04_harness_agent_mode_isolated`: integration-marked, skipped when the CLIs are missing. The real-home guard must stay unchanged.

## 5. Not in G2

- The proxy itself (G3).
- CI wiring for live agent sessions. That's FR-PROOF-04's CI part, which needs CI secrets policy decisions.
- Any change to the package.
