# G5 spec: media policy, the ChatGPT-login upstream, connection reuse

Oct 20, 2026 · status: **approved by Josh, Oct 20, 2026** · requirements: FR-GATE-07, FR-GATE-06 (connection reuse, the ChatGPT upstream), NFR-01. Builds on `docs/gate-parser-spec.md` (class M, decision 9), `docs/gate-transport.md` (T5, CX2) and `docs/gate-scan-spec.md`.

Response streaming (the rest of FR-GATE-06) already shipped in G3 and is covered by `test_fr_gate_06_relay.py` and `test_fr_gate_06_proxy.py`. G5 doesn't change the relay.

## Part 1: media (FR-GATE-07)

**Today:** `walk_common.is_media` skips any string that starts with `data:` and has `;base64,` in its first 200 characters. It does this in every field, including free text. Anthropic `base64` sources and Responses `file_data` / `input_audio.data` are skipped by name. Nothing is logged.

**Two holes this leaves:**
1. **Prefix bypass.** A text field that starts like a data URL is never scanned, even when the tail after the base64 is plain text. For example, a `tool_result` of a file whose first line is a data URL, followed by `OPENAI_API_KEY=...`.
2. **Text sent as a file.** `input_file.file_data` and an Anthropic `document` with a base64 source can carry a text file: `.env`, JSON, source code. Skipping it by name lets a secret through without scanning.

**Policy:**

1. **Media is decided by position, not content.** A value is M only in the fields the parser tables mark M:
   - Anthropic `image.source.data` and `document.source.data` when the source type is `base64`
   - Responses `input_image.image_url` and `computer_call_output.output.image_url` when it is a `data:` URL
   - `input_file.file_data`
   - `input_audio.data`
   - `image_generation_call.result`

   Everywhere else a `data:` string is ordinary text and is scanned. `is_media` is removed from `leaf()`.
2. **The payload must be strict base64.** An M value must be `[A-Za-z0-9+/]*={0,2}` to its end, after the `data:<type>;base64,` prefix if there is one. Otherwise the request blocks with a new reason, `MEDIA_MALFORMED`. Base64 can't hide a readable tail, so a strict payload can't smuggle text past the gate.
3. **Images, PDFs and audio pass with a warning (the v1 default).** The bytes are forwarded unchanged and aren't decoded or scanned. The warning is a request-log field, `media`, holding a list of `{kind, bytes}`:
   - `kind` comes from a fixed list: `image/png`, `image/jpeg`, `image/gif`, `image/webp`, `application/pdf`, `audio`, `other`. It never holds the raw media type string.
   - `bytes` is the decoded size.

   This gives the warning no path for content. OCR stays cut (design doc cut order, item 3).
4. **Text-type files are decoded and scanned.** Text types are `text/*`, `application/json`, `application/xml`, `application/x-yaml`, `application/toml`, a `file_data` whose `filename` has a text extension, and a payload that decodes to valid UTF-8 with no NUL bytes in its first 8 KiB.
   - A finding blocks with `MEDIA_SECRET`. The gate doesn't rewrite inside base64. Re-encoding would change the file's length and its hash, and it's rare enough not to be worth it in v1.
   - The block message tells the agent to send the file as text instead, where it can be redacted.
   - Decoded bytes count toward the 64 MiB cap (decision 9).
   - Scan results are cached the same way G4 caches text: by SHA-256 of the decoded bytes.

**Trade-off note (rule 4):** rule 4 doesn't change a threshold, but two of these rules change what reaches the provider:
- Rule 2 can block a request the provider would accept. For example, base64 with line breaks; RFC 2045 allows them. Whether any agent sends them is checked against the G2 captures before coding.
- Rule 4 blocks text files that hold a secret, where today they pass unscanned.

Both choices fail closed.

## Part 2: the ChatGPT-login upstream

**Signal (T5, VERIFIED):** Codex under route A sends every request to the gate's `/v1/...` path, whatever its login. Only the `chatgpt-account-id` header tells a ChatGPT login apart.

**Rule:**
- An `openai.*` route that carries `chatgpt-account-id` goes to the ChatGPT upstream:
  - host: config `chatgpt_upstream`, default `https://chatgpt.com`
  - path: the `/v1` prefix replaced with `/backend-api/codex`, so `POST /v1/responses` becomes `POST /backend-api/codex/responses` and the query string is kept
- Without the header, the request goes to `openai_upstream` unchanged.
- An Anthropic route never looks at the header.
- The prefix is a constant, not configurable. `parse_upstream` still refuses a path in any upstream URL.

**Unverified:** `GET /v1/models` under a ChatGPT login. CX2 doesn't say whether `/backend-api/codex/models` exists. In v1 it is mapped like the others. If the live check below shows a 404, Codex should behave as it does against the mock in T5 (the turn completed despite 404s), and the spec gets a line.

**Not affected:** the `chatgpt_base_url` calls (`/backend-api/wham/...`, plugins, analytics). Codex sends these straight to chatgpt.com, not through the provider base URL (T5). They carry account checks, not prompt content. They stay outside the gate, which matches the transport notes.

**Safety:** a wrong upstream choice can only cause a 401 from the other OpenAI host. Both hosts are OpenAI's, and every body is scanned before either one sees it. Headers pass byte-identical (FR-INSTALL-05). Of all the headers, only `chatgpt-account-id` is read here, and only for whether it is present.

## Part 3: upstream connection reuse (NFR-01)

**Today:** every request opens a new TCP and TLS connection, which is closed after the reply. On a real network a TLS handshake to the provider costs a round trip or two, tens to hundreds of ms on each turn. The agent's own direct connection doesn't pay it. That cost lands on NFR-01's added latency but isn't in the G4 number, because G4 measured only the scan.

**Design (stdlib `http.client`, no new dependency):**
- **The pool.** One idle-connection pool per `Upstream` `(scheme, host, port)` behind a lock.
  - Last in, first out.
  - At most 4 idle connections per upstream.
  - An idle connection is discarded after 30 s.
- **Returning a connection** is allowed only when all of these hold:
  - the response was read to its framed end (Content-Length met, or the terminating chunk was seen)
  - the reply was HTTP/1.1
  - there was no `Connection: close`
  - there was no error or timeout
  - the agent didn't hang up mid-stream

  Every other case closes the connection: errors, drops, cut-off streams and close-delimited bodies.
- **Stale connections.** If a *reused* connection fails before the first response byte (reset, broken pipe, `RemoteDisconnected`), the same scanned bytes are sent once more on a new connection. A new connection that fails gets a 502, as it does today.
  - **Trade-off:** if the provider read the request and then dropped the connection without answering, it sees the request twice. That can cost one duplicate generation, and the agent only receives one reply. This is what HTTP client libraries generally do for an idle connection.
  - **No security effect:** the retry resends bytes that were already scanned, to the same host.
- **Credentials.** Connections carry no credentials (auth is a header on each request), so one pool may serve several sessions. The ChatGPT and API hosts are different `Upstream`s, so they never share a connection.
- **Closing.** `GateServer.close()` closes every pooled connection.
- **Logging.** The request log gains `upstream_reused: bool` and `ms_connect`, which is 0 on reuse.

## Test plan, tests first

**Media**
- `test_fr_gate_07_image_and_pdf_pass_with_warning`: forwarded byte-identical; the log `media` field has the kind and size and nothing else
- `test_fr_gate_07_data_url_in_text_field_is_scanned`: the prefix-bypass case; the canary is redacted
- `test_fr_gate_07_malformed_base64_media_blocks`
- `test_fr_gate_07_text_file_with_secret_blocks` (Anthropic `document` and Responses `input_file`), and `..._clean_text_file_passes`
- `test_fr_gate_07_media_kind_is_from_fixed_list`: a crafted `media_type` never reaches the log

**ChatGPT upstream**
- `test_fr_gate_06_chatgpt_account_routes_to_chatgpt_upstream`: against the mock, the recorded path is `/backend-api/codex/responses` and the headers are byte-identical
- `test_fr_gate_06_api_key_request_routes_to_openai_upstream`
- `test_fr_gate_06_chatgpt_upstream_url_with_path_refused`

**Connection reuse**
- `test_fr_gate_06_second_request_reuses_upstream_connection`: the mock counts accepted connections
- `test_fr_gate_06_cut_stream_connection_not_reused`, `..._connection_close_not_reused` and `..._error_not_reused`
- `test_fr_gate_06_stale_pooled_connection_retried_once`: the mock closes idle connections; the request succeeds and the mock gets it once
- `test_fr_gate_06_fresh_connection_failure_not_retried`: a 502, and the mock gets it at most once

**End to end (rule 7)**
- **Codex.** A real `codex exec` session in a throwaway home, with the fake ChatGPT `auth.json` from `bench/g2_probes.py`, goes through the gate. The gate's `chatgpt_upstream` points at the mock. Pass condition: the mock records `/backend-api/codex/responses` and 0 known-format canaries.
- **Claude Code.** A real `claude -p` session sends an image and a text file through the gate to the mock. Pass condition: the image is forwarded byte-identical with a `media` log entry, and the text file holding a canary is blocked.
- **Latency.** `bench/gate_latency.py` on replayed G2 sessions over TLS to the mock (throwaway certificate), with the pool on and off. Added p95 is published in `bench/results/gate/`.

**Live check (Josh runs it, needs a real ChatGPT login):** one Codex turn through the gate to the real `chatgpt.com`, which confirms the path mapping and settles `/models`. No canary goes in that prompt. The result goes into `docs/gate-transport.md` as T10.

## Not in G5

- OCR of images
- rewriting secrets inside base64 text files
- the audit log (G6), which takes over the `media` warning
- install adapters (G7)
- WebSocket support (route B stays refused with 426)

## Decisions

Josh approved the spec as written on Oct 20, 2026, without picking an answer to the open questions. The defaults below were taken, the way G4's were.

1. **A base64 text file with a secret is blocked.** It is not decoded, redacted and re-encoded.
2. **Media must be strict base64.** Base64 with line breaks blocks. The G2 captures are checked for line breaks before coding. If an agent sends them, this comes back to Josh before the code changes.
3. **The live ChatGPT check runs before G5 closes.** It is in the test plan above, and Josh runs it with his own login.
