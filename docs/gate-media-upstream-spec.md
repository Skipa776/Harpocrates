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
   - `kind` comes from a fixed list: `image/png`, `image/jpeg`, `image/gif`, `image/webp`, `application/pdf`, `audio`. It never holds the raw media type string (see the implementation notes: the kind is read from the payload's signature).
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

**Implementation notes (Oct 20, 2026, after review):** the code decides "binary or text" by content, not by label. This is stricter than rule 4 as written.
- **Binary means a known signature.** A payload passes unscanned only when its first bytes are a known format: PNG, JPEG, GIF, WEBP, PDF, or audio (WAV, MP3, Ogg, FLAC). Its kind in the log comes from those bytes, never from the request's label, so no request text can reach the note. The list therefore has no `other`.
- **Everything else is read as text and scanned.** That includes a text file with any label, UTF-16 (read both ways when the bytes hold a NUL), text with a stray NUL or invalid byte, and an unknown binary format. A secret in any reading blocks with `MEDIA_SECRET`. The security review found that the first version let UTF-16 text and text with one bad byte through as "binary", which this closes.
- **Trade-off (rule 4):** an unknown binary format, such as a zip or a `.docx`, is scanned as replaced text, and a false positive blocks it. A payload that starts with a real image signature and has text appended still passes unscanned. That is accepted for v1, like the rest of an image's content.
- **Because the content decides,** there is no filename-extension rule and no list of text media types.
- **Caching.** Decoded text is scanned through the finder's existing per-session cache. The cache is keyed by SHA-256 of the text, so an unchanged file is not rescanned on every turn.
- **No separate size check.** Decoded base64 is always smaller than the base64 text, and that text is already inside the capped body.
- **Other members of an Anthropic base64 source,** apart from `type`, `media_type` and `data`, are scanned as text. Before G5 the whole source was skipped.
- **A base64 data URL in a `url`-type source** is media. Any other URL there is scanned as text.

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

## Implementation notes and results for parts 2 and 3 (Oct 20, 2026)

**Part 2, the ChatGPT upstream.**
- **Config.** `GateConfig.chatgpt_upstream` defaults to `https://chatgpt.com` and is checked like every upstream: scheme, host and port only. The prefix is the constant `CHATGPT_PATH_PREFIX`.
- **Routing.** `GateHandler._destination` picks the upstream. It reads only whether a `chatgpt-account-id` header is present, in any case, and never its value.
- **Tests.** `cli/tests/test_fr_gate_06_chatgpt_upstream.py`, 13 tests.
- **End to end (rule 7).** Suite `g5`, report `bench/results/proof/20261020-gate-e2e-g5k.json`. Real `codex exec` (0.163.0) runs with the fake ChatGPT login from `bench/g2_probes.py`, and the gate's ChatGPT upstream is the mock.
  - All 4 model requests arrive on `/backend-api/codex/responses` and none on `/v1`.
  - The three scripted reads complete, and 0 known-format canaries are received (10 placeholders).
  - The gate-off control leaks 5. The mock answers the ChatGPT paths as aliases of the OpenAI routes.
- **Still open.** The live check against the real chatgpt.com (T12, needs Josh's login) is the last open item of part 2.

**Part 3, connection reuse.**
- **Code.** `UpstreamPool` in `cli/Harpocrates/gate/upstream.py`, owned by the gate server and closed with it. `open_upstream` still opens one new connection and is kept for direct use.
- **When a connection is kept.** Only when all of these hold:
  - the relay ended `complete`;
  - the response object is closed, so its body was read to the framed end;
  - the reply was HTTP/1.1;
  - `will_close` is false.
- **Bodiless replies are not kept.** A 204 or 304 reply, a reply to HEAD and an informational reply are relayed without reading a body, so they aren't kept, which is the simple choice.
- **Known limit, not new.** `http.client` skips only `100 Continue`. A `103 Early Hints` followed by the real reply reaches the agent as a final 103, and the real reply is lost. That connection is closed, not pooled, so nothing crosses requests.
- **Routing.** An agent that adds `chatgpt-account-id` to an API-key request sends that key to chatgpt.com. Both hosts are OpenAI's, the body is scanned first, and `routes.classify` lets only `/v1/responses` and `/v1/models` through, so no other path can be rewritten.
- **Readable while idle (from review).** An idle connection is reused only if nothing is waiting to be read on it, checked with `select` and pending TLS bytes. A provider that closed it, sent an idle-timeout notice (a 408) or any stray reply makes it readable. It is then closed, and its bytes are never read as the reply to the next request, which may belong to another agent session. This also catches most idle closes before the send, so they need no retry.
- **Stale connections, narrowed (from review).** A reused connection is retried once on a new connection only:
  - for a reset, broken pipe or TLS EOF while *sending*;
  - or for a close with **no** reply byte (`RemoteDisconnected`) while reading the reply.

  Anything later is a 502 with no retry: a reset partway through the reply head, or a timeout. Either could follow a processed request.
- **Log fields.** The request log has `upstream_reused` and `ms_connect`.
- **Reuse with real agents.** In the `g5k` run, every request after a session's first went out on a pooled connection: 2 to 5 per session, across all ten Claude Code and Codex sessions. Nothing else in those runs changed.
- **Tests.** `cli/tests/test_fr_gate_06_connection_reuse.py`, 18 tests.
- **Latency.** `python -m bench.gate_latency [--tls] [--no-pool]`, pass-through scan, 100 KB body, 200 calls. Reports are `bench/results/gate/latency-2026-10-21{,-nopool,-tls,-tls-nopool}.json`, dated in UTC.

  | Upstream | Pool off, added p50 / p95 | Pool on, added p50 / p95 |
  | --- | --- | --- |
  | TLS (throwaway cert) | 1.14 / 1.31 ms | 0.19 / 0.24 ms |
  | Plain HTTP | 0.30 / 0.93 ms | 0.19 / 0.21 ms |

  On loopback the saving is about one TLS handshake, 1 ms. Over a real network a handshake costs one or two round trips more, so the saving there is larger. These runs used the synthetic 100 KB body of the G3 reference, not the G2 replay the test plan named. The pool changes only the transport, which `bench/gate_scan_latency.py` (the G2 replay) does not measure.

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

**End-to-end result for media (Oct 20, 2026; suite `g5`, report `bench/results/proof/20261020-gate-e2e-g5g.json`):** real `claude -p` (2.1.304) and `codex exec` (0.163.0) sessions ran through the gate to the mock, each also with the gate off as a control. Each agent read a real PNG and the canary `.env` saved as `notes.png`, first with its own image tool and then through a test MCP server (`bench/mcp_image_server.py`) that returns any file as an image or as a PDF resource.
- **The real PNG.** It reached the mock byte-identical in every run (`png.identical` equals `png.payloads` in the report) and was logged as `image/png` in each request that carried it.
- **The agents' own image tools and MCP images.** Neither agent sent the text file as base64 this way (T11): both check the bytes, so nothing was blocked and 0 canaries were received, with the gate on or off.
- **The MCP PDF resource through Codex is the one path that sent the file.** With the gate off, the mock received all 5 canaries from the `.env` (3 known-format) inside base64, but in a text field: Codex sends the resource as JSON text with the base64 blob in it. With the gate on, 0 were received at any base64 alignment. The whole blob was replaced by one `generic` placeholder. That is most likely the detector's entropy stage, since the gate log does not record which detector stage fired. The gate did not decode the blob.
- **Open gap, closed by FR-GATE-11 (`docs/gate-base64-text-spec.md`, run `g5i`): the 0 above was accidental protection, not a proven one.**
  - **It isn't caught on purpose.** A secret base64-encoded inside a text field is caught only when the encoded run looks random to the detector, and a short one may not.
  - **Benign blobs get redacted.** A benign blob in text, such as a real PDF resource, gets redacted the same way.
  - **Proposed fix (not in this spec):** decode long base64 runs inside R and S text, scan the decoded text, and block on a finding. It needs its own decision.
- **What is not proven.** The `MEDIA_SECRET` block itself has not been reached by an agent in these versions. It is proven through the real gate with a scripted client (`test_fr_gate_07_text_file_with_secret_is_blocked_through_the_gate`).
- **Detector false positives seen.** One or two placeholders per gate-on run were not canaries: a temp-dir file path in Claude Code's image error, and a line of Codex's own system prompt. Both are the entropy-on-paths and identifiers weakness (M2 plan, D4). Neither touched a media payload.

**Live check (Josh runs it, needs a real ChatGPT login):** one Codex turn through the gate to the real `chatgpt.com`, which confirms the path mapping and settles `/models`. No canary goes in that prompt. The result goes into `docs/gate-transport.md` as T12 (T10 and T11 are taken).

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
