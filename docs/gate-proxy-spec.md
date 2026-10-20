# G3 spec: the localhost proxy

Oct 19, 2026 · status: **awaiting Josh's approval** · requirements: FR-GATE-01, SEC-03, FR-INSTALL-04, FR-INSTALL-05, plus the transport parts of FR-GATE-03 and FR-GATE-06. Builds on `docs/gate-parser-spec.md` (decisions 3, 9 to 13, 15) and `docs/gate-transport.md` (G2 results).

G3 is the plumbing: accept, check, decode, hand to a scan step, forward, stream back. Scanning and redaction are G4. **G3 never forwards an unscanned request.** Until G4 lands, the scan step is a stub that raises, so every request is blocked (fail closed). Tests swap in a pass-through scan step explicitly.

## Where and how

| Item | Choice |
| --- | --- |
| Location | `cli/Harpocrates/gate/`, one module per concern, on the docstring enforce list |
| Run | `python -m Harpocrates.gate --port N --token-file F`. `harpo guard up` (FR-UX-01, G7) wraps it later |
| Server | Standard library: `http.server.ThreadingHTTPServer`, bound to `127.0.0.1` only; refuses to start on any other address |
| Upstream client | Standard library: `http.client.HTTPSConnection` with the system trust store and certificate verification always on. No new dependency |
| Upstream hosts | `api.anthropic.com` for Anthropic routes and `api.openai.com` for OpenAI routes by default. Overridable per provider in the gate config; only loopback overrides may use plain HTTP (tests, mock) |

## Inputs and checks, in order

1. **Token (SEC-03).** Header `x-harpo-gate-token` must equal the session token: a new random token on every start, read from a `0600` file. The check uses a constant-time compare. Missing or wrong → `401`, nothing forwarded. The token is removed before forwarding.
2. **Route.**

   | Route | Treatment |
   | --- | --- |
   | `POST /v1/messages`, `POST /v1/messages/count_tokens`, `POST /v1/responses` | Go to the scan step |
   | `GET /v1/models` | Passed through (no body) |
   | WebSocket upgrade | `426` (decision 3) |
   | Anything else | Blocked, `400` (decision 13) |

3. **Body (decision 9, FR-INSTALL-04).** Read by `Content-Length` or chunked, at most 64 MiB after decoding.
   - **Encodings:** `identity` and `gzip` are decoded. `zstd` is decoded when a zstd module is available, otherwise blocked. Any other `Content-Encoding` is blocked.
   - **Malformed input:** a body over the cap, a truncated body or a malformed chunk is blocked. JSON strictness (decision 15) is enforced by the parser in G4.
4. **Headers (FR-INSTALL-05).** Auth headers (`x-api-key`, `Authorization`, `anthropic-version`, `anthropic-beta`, `ChatGPT-Account-ID` and the others named in the transport notes) are forwarded byte-identical and never logged. Hop-by-hop headers are dropped as HTTP requires. Scanning non-auth header values (decision 12) is G4.
5. **Scan step.** Called with the decoded body and the route. It returns either the bytes to forward or a block reason. A timeout (budget in the gate config, default 5 s) or any exception → block (FR-GATE-03).

## Outputs

- **Forwarded request.** The body is re-encoded as received (identity when it arrived as identity). Upstream `Content-Length` is set from the forwarded bytes.
- **Response (FR-GATE-06).** Status, headers and body are streamed back unchanged as they arrive, including SSE, with no buffering of streamed bodies. An upstream connection failure or timeout before any response byte → `502` in the provider's error shape. After bytes have started, the connection is closed (the agent sees a truncated stream, never altered content).
- **Blocks.** HTTP `400` (`401` for the token, `426` for upgrades) in the provider's own error shape. The message is plain and avoids the recovery-trigger phrases (decision 10).
- **Log.** One JSONL line per request: time, route, status, decision, sizes and timings. Never a header value, never body content. The full audit log is FR-LOG-01 (G6).

## Failure behavior summary

Every error path blocks: token, route, body, encoding, scan error and scan timeout. The only exceptions are upstream failures, which can't forward anyway. There is no override flag in G3. The logged override of FR-GATE-03 comes with G4, after you approve its wording.

## Test plan, tests first, using `bench/mock_provider.py` as upstream

- `test_fr_gate_01_forwards_to_upstream_and_streams_back`, for JSON and SSE, in both formats.
- `test_fr_gate_01_binds_loopback_only`
- `test_sec_03_missing_or_wrong_token_rejected`
- `test_sec_03_token_not_forwarded`
- `test_fr_install_05_auth_headers_forwarded_byte_identical_and_never_logged`
- `test_fr_install_04_gzip_decoded`
- `test_fr_install_04_unknown_encoding_blocked`
- `test_fr_gate_03_scan_stub_blocks_everything`: the default state, proving G3 alone never forwards unscanned.
- `test_fr_gate_03_scan_error_and_timeout_block`
- `test_fr_gate_01_unknown_route_blocked_and_websocket_426`
- `test_fr_gate_01_oversized_and_truncated_bodies_blocked`
- `test_fr_gate_06_upstream_failure_502`
- `test_fr_gate_06_mid_stream_drop_closes`

Plus one timing record: added latency against direct-to-mock for a 100 KB body, p50 and p95 over 200 calls, kept in `bench/results/gate/`. This is a reference for NFR-01 (M3), not a gate on G3.

## Not in G3

Parsing, scanning, redaction, placeholders, determinism and thinking-block dropping (G4). Images and PDFs (G5). Audit log and SEC-01 grep (G6). Install adapters, `harpo guard up` and `harpo doctor` (G7). SEC-10 (proposed, pending your approval).
