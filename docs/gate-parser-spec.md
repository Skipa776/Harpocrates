# Gate request parser spec (FR-GATE-02)

Oct 19, 2026 · status: **decisions approved; G2 capture results added below ("To verify"). No gate code exists yet.** M2, step G1. Covers what `harpo-gate` does with the body of every request, per provider format. Transport (paths, headers, SSE, WebSocket) is in [gate-transport.md](gate-transport.md).

Formats in v1: Anthropic Messages (`POST /v1/messages`, Claude Code) and OpenAI Responses (`POST /v1/responses`, Codex). OpenAI Chat Completions comes later, with the Copilot adapter; it gets its own section here then.

All external facts cite a source from the table at the end, each checked 2026-10-19. A claim is **VERIFIED** (source named) or **TO VERIFY** (what will confirm it). Both lists are collected in the last section.

## Decisions (approved by Josh, Oct 19, 2026)

These settle the open questions below and bind the implementation.

1. **`previous_response_id`:** block in v1. Route A (custom provider) is not expected to send it (V5). A relay that tees response ids can come later.
2. **JSON-in-string tool arguments:** parse, redact string leaves, splice the redacted strings back into the raw bytes. A clean body is forwarded byte-identical. A secret in a free-form object key blocks the request.
3. **WebSocket:** not in v1. The gate refuses an upgrade with 426, and Codex falls back to HTTP.
4. **A secret in provider-signed readable text** (`reasoning.summary`, `web_search_result` title or URL, `cited_text`): block, until a live test (V2) shows the provider accepts a rewrite.
5. **FR-GATE-10:** drop every thinking and redacted-thinking block from the request instead of blocking it. The API allows this, nothing unredacted is forwarded, and the session survives. `design.md` FR-GATE-10 is reworded to match.
6. **A credential the user configured for a provider-hosted MCP connector** (`authorization_token` and similar): pass, with a logged warning that names the field, never the value.
7. **Session key:** in memory only (SEC-02). A gate restart breaks prompt caching and thinking for sessions already open. That is accepted and documented.
8. **Redacting a newly found secret in earlier turns:** off in v1, because it would change bytes the provider already saw.
9. **Body cap:** 64 MiB, applied after decompression. v1 buffers the whole body before forwarding.
10. **Block response:** HTTP 400 in the provider's own error shape. The message avoids the phrases Claude Code uses to trigger its own recovery ("thinking", "signature", "bound to a different conversation", "Extra inputs are not permitted", "prompt is too long").
11. **Gate token (SEC-03):** a new token on every `harpo guard up`, written into `ANTHROPIC_CUSTOM_HEADERS` and Codex `http_headers`.
12. **Non-auth header values:** scanned. A finding blocks the request.
13. **Unknown routes:** block (fail closed). The Codex features this disables are listed in `docs/gate-transport.md`.
14. **`design.md` Codex row:** corrected (zstd and WebSocket apply only to the built-in OpenAI provider).
15. **Strict parsing:** reject duplicate JSON keys, invalid UTF-8 and nesting deeper than 128.

## Goal and the one rule

Every string the model will see goes through the scanner. Anything found is replaced by a placeholder `<<HARPO:type:xxxx>>`. Anything the gate cannot read or cannot rewrite is either passed untouched (when it holds no readable text) or blocks the request. Nothing is ever silently skipped.

The rule that makes this safe: **the block types are a closed list, the fields are an open list.** An unknown block type blocks the request (FR-GATE-03). An unknown string field on a known block type is scanned like its neighbours. Only the fields named as identifiers, opaque or media below are skipped.

## Field classes

| Class | Meaning | Gate action |
| --- | --- | --- |
| **R** | Text written by the client (user, agent, tool) | Scan, rewrite in place |
| **S** | Readable text inside a provider-produced block that the provider signs or pairs with ciphertext | Scan, never rewrite. A finding blocks the request (FR-GATE-10) |
| **I** | Identifier or enum: `type`, `role`, `id`, `*_id`, `name`, `signature`, `model`, `status` | Not scanned, never changed. Rewriting an id breaks tool-call pairing |
| **O** | Opaque ciphertext or signature: `signature`, `redacted_thinking.data`, `encrypted_content`, `encrypted_index`, `encrypted_stdout` | Not scanned (the gate cannot read it), never changed |
| **M** | Binary media: base64 image, base64 PDF, `file_data` | Pass with a warning in the audit log (FR-GATE-07) |
| **K** | Credential the user sent on purpose to a provider-hosted connector: `mcp_servers[].authorization_token`, Responses `mcp` tool `authorization` and `headers` | Not scanned, never logged, pass with a warning. Open question 4 |

## Anthropic Messages (`POST /v1/messages`)

Body shape (all VERIFIED, A1): `model`, `max_tokens`, `messages[]` (`role`, `content` as a string or an array of blocks), `system` (string or array of text blocks), `tools[]`, `tool_choice`, `thinking`, `metadata`, `stop_sequences`, `cache_control` (top level), `container`, `output_config`, `mcp_servers`, `stream`, and more. The list grows with each release. `POST /v1/messages/count_tokens` takes the same content fields and is parsed by the same code (A6).

### Top-level fields

| Field | Class | Policy |
| --- | --- | --- |
| `system` string | R | Scan, rewrite. Keep it a string; never convert string and array into each other |
| `system` blocks | R | Per text block. Block order, count and `cache_control` unchanged. Claude Code puts an attribution block first and the API strips it by position, so it must stay its own first block, byte for byte (C1) |
| `metadata.user_id`, `stop_sequences[]` | R | Scan, rewrite |
| `output_config.format.schema` | R | Scan every string leaf (descriptions, enums, defaults) |
| `compaction.instructions` | R | Scan, rewrite |
| `mcp_servers[]` | `name`, `url`: R. `authorization_token`: K | See open question 4 |
| `container`, `thinking`, `tool_choice`, `context_management`, `service_tier`, `speed`, `inference_geo`, numbers, booleans | I | Not scanned |
| `cache_control` (top level, on blocks, on tools) | I | Never touched. See determinism below |
| Any other top-level string field | R | Scanned |

### Content blocks

`messages[].content` as a plain string is the same as one `text` block: scan it, rewrite it, keep it a string.

| Block type | Scan (class) | Rewrite or policy | Test |
| --- | --- | --- | --- |
| `text` | `text` (R), `citations[].cited_text` (R) | In place. `cache_control`, `citations` structure unchanged | `test_fr_gate_02_anthropic_text` |
| `image` | none (M) | Base64, `url` and `file` sources pass with a warning. A `url` string is scanned as R (it may hold a token) | `test_fr_gate_02_anthropic_image` |
| `document`, source `text` (`text/plain`) | `source.data` (R), `title` (R), `context` (R) | In place. The data is plain text, not base64 | `test_fr_gate_02_anthropic_document_text` |
| `document`, source `base64` (PDF) | `title`, `context` (R). Data (M) | Pass with a warning | `test_fr_gate_02_anthropic_document_pdf` |
| `document`, source `content` | Nested `text` and `image` blocks, handled as above | Recurse | `test_fr_gate_02_anthropic_document_content` |
| `document`, source `url` or `file` | `url` string (R) | Pass with a warning: the provider fetches the bytes, the gate cannot | `test_fr_gate_02_anthropic_document_url` |
| `search_result` | `content[].text`, `title`, `source` (R) | In place | `test_fr_gate_02_anthropic_search_result` |
| `tool_use` | `input` (R, JSON object) | Rewrite string leaves, see JSON rule. `id`, `name`, `caller` are I | `test_fr_gate_02_anthropic_tool_use` |
| `tool_result`, `content` string | the string (R) | In place | `test_fr_gate_02_anthropic_tool_result_string` |
| `tool_result`, `content` blocks | Nested `text`, `image`, `search_result`, `document`, `tool_reference` (name only, I), `browser_state` (`tabs[].title`, `tabs[].url`: R) | Recurse | `test_fr_gate_02_anthropic_tool_result_blocks` |
| `thinking` | `thinking` text (S), `signature` (O) | A finding blocks the request. Unchanged otherwise | `test_fr_gate_02_anthropic_thinking` |
| `redacted_thinking` | `data` (O) | Pass | `test_fr_gate_02_anthropic_redacted_thinking` |
| `compaction` | `content` summary (S), `signature` (O) | A finding blocks. The API rejects an edited summary (A4) | `test_fr_gate_02_anthropic_compaction` |
| `server_tool_use` | `input` (R) | Rewrite string leaves | `test_fr_gate_02_anthropic_server_tool_use` |
| `web_search_tool_result` | `title`, `url` (S). `encrypted_content` (O) | A finding blocks (conservative, open question 3) | `test_fr_gate_02_anthropic_web_search_tool_result` |
| `web_fetch_tool_result` | Nested `document` block, `url` | Recurse as `document` | `test_fr_gate_02_anthropic_web_fetch_tool_result` |
| `code_execution_tool_result`, `bash_code_execution_tool_result` | `stdout`, `stderr` (R). `encrypted_stdout` (O) | In place on the readable fields | `test_fr_gate_02_anthropic_code_execution_tool_result` |
| `text_editor_code_execution_tool_result` | `view_result.content`, `str_replace_result` line text (R) | In place | `test_fr_gate_02_anthropic_text_editor_tool_result` |
| `tool_search_tool_result` | `tool_references[].tool_name` (I) | Pass | `test_fr_gate_02_anthropic_tool_search_tool_result` |
| `mcp_tool_use` | `input` (R). `server_name`, `name` (I) | Rewrite string leaves | `test_fr_gate_02_anthropic_mcp_tool_use` |
| `mcp_tool_result` | `content[].text` (R) | In place | `test_fr_gate_02_anthropic_mcp_tool_result` |
| `mcp_tool_listing` | `tools[].description`, `input_schema` strings (S) | Echoed from the provider, sent back unchanged (A5). A finding blocks | `test_fr_gate_02_anthropic_mcp_tool_listing` |
| `container_upload` | `file_id` (I) | Pass | `test_fr_gate_02_anthropic_container_upload` |
| any other `type` | n/a | **Block** (FR-GATE-03) | `test_fr_gate_02_anthropic_unknown_block_blocked` |

`messages[]` entries with `role: "system"` (mid-conversation system messages, A3) hold `text`, `tool_addition` and `tool_removal` blocks:

| Block type | Scan (class) | Rewrite or policy | Test |
| --- | --- | --- | --- |
| `role: "system"` `text` | R | In place. The message stays where the caller put it; moving it into `system` invalidates thinking (A2) | `test_fr_gate_02_anthropic_system_role_text` |
| `tool_addition` with `tool_definition` | `definition.description`, `input_schema` strings (R) | In place. Same rules as `tools[]` | `test_fr_gate_02_anthropic_tool_addition` |
| `tool_removal`, `tool_reference` | names (I) | Pass | `test_fr_gate_02_anthropic_tool_removal` |

`tools[]` entries: a custom tool has `name` (I), `description` (R), `input_schema` (R, string leaves), `input_examples` (R, string leaves). Typed server tools (`bash_*`, `code_execution_*`, `tool_search_tool_*`, `browser_toolset_*`) and `mcp_toolset` hold no free text and pass. An unknown tool `type` blocks. Tests: `test_fr_gate_02_anthropic_tools_custom`, `test_fr_gate_02_anthropic_tools_server`, `test_fr_gate_02_anthropic_tools_mcp_toolset`, `test_fr_gate_02_anthropic_tools_unknown_type_blocked`. All VERIFIED against A1 and A5 for the shapes above; the full union is not listed in one place, so the corpus capture in G2 is the check (V6, result in "To verify": only some of these shapes were sent by the agents).

### Why thinking matters here

The API checks every signed thinking block against the exact `system`, `tools` and `messages` that came before it. Editing an earlier message, a tool result or a tool description makes that block and every later one invalid (A2, VERIFIED). A gateway rewrite counts as an edit.

Redaction is still safe, for one reason: the signature binds to what the provider received, which is the redacted bytes. Next turn the agent resends the original, the gate redacts it to the same bytes, and the prefix matches. This fails only if redaction of the same text ever comes out different. See the determinism rules.

If it does fail, Claude Code removes the thinking blocks and retries (C1, VERIFIED). The conversation survives, with a cache miss and no old reasoning.

## OpenAI Responses (`POST /v1/responses`)

Body shape (VERIFIED, O1): `model`, `input` (string or array of items), `instructions`, `previous_response_id`, `conversation`, `store`, `include`, `tools`, `tool_choice`, `reasoning`, `text`, `metadata`, `prompt` (`id`, `variables`), `prompt_cache_key`, `stream`, and more. Codex also sends `client_metadata` (a string map) and `access_programs` (CX7, VERIFIED). Codex builds the full `input` each turn, sends `store: false`, and its HTTP body has no `previous_response_id` field (CX7, VERIFIED).

### Top-level fields

| Field | Class | Policy |
| --- | --- | --- |
| `instructions`, `input` string | R | Scan, rewrite |
| `prompt.variables` (strings), `metadata` values, `client_metadata` values, `user`, `safety_identifier`, `prompt_cache_key` | R | Scan, rewrite. `prompt_cache_key` must stay stable per session; redaction is deterministic so it does |
| `text.format` (`schema`, `name`, `description`) | R | Scan string leaves |
| `previous_response_id` | I | See the next section |
| `conversation` | I | **Block in v1** (server-side history the gate cannot see) |
| `include`, `reasoning`, `store`, `stream`, `service_tier`, numbers, booleans | I | Not scanned |
| Any other top-level string field | R | Scanned |

### `previous_response_id` and server-side history

The gate sees only what is in the request. With `previous_response_id` (or `conversation`) the provider adds history the gate never saw in this request. That history is safe only if every earlier request in the chain passed through the gate, because then it was redacted when sent.

**Proposed v1 policy.** The gate records every `response.id` it relays (a read-only tee of the stream; bytes to the agent stay unchanged, FR-GATE-06). A request whose `previous_response_id` is in that set passes, and only its new `input` items are scanned. Any other value blocks with a clear message. After a gate restart the set is empty, so a continuation blocks and the agent must send a full request. This needs a decision (open question 1).

Codex over HTTP never sends it, so FR-GATE-04 holds in full there. Codex over WebSocket sends only new items plus `previous_response_id` (CX1, O2, VERIFIED), so the gate scans each item once, when it is first sent. Tests: `test_fr_gate_02_responses_previous_response_id_seen_passes`, `test_fr_gate_02_responses_previous_response_id_unseen_blocked`, `test_fr_gate_02_responses_conversation_blocked`.

### `input` items

A bare string `input` is one user message. Message items come in two forms (O1): the short form with `content` as a string, and the long form with typed content parts. Roles: `user`, `assistant`, `system`, `developer`.

| Item or part | Scan (class) | Rewrite or policy | Test |
| --- | --- | --- | --- |
| `message` content string | R | In place | `test_fr_gate_02_responses_message_string` |
| part `input_text` | `text` (R) | In place | `test_fr_gate_02_responses_input_text` |
| part `output_text` | `text` (R), `annotations[]` `title`, `url`, `filename` (R), `logprobs[].token` (R) | In place. Echoed model output, handled as R because it carries no signature | `test_fr_gate_02_responses_output_text` |
| part `refusal` | `refusal` (R) | In place | `test_fr_gate_02_responses_refusal` |
| part `input_image` | `image_url` (M when a `data:` URL; R otherwise), `file_id` (I) | Pass with a warning | `test_fr_gate_02_responses_input_image` |
| part `input_file` | `filename` (R). `file_data` (M). `file_url` (R) | Pass with a warning. Text files sent as base64 are not decoded in v1 | `test_fr_gate_02_responses_input_file` |
| `function_call` | `arguments` (R, JSON string), `name`, `namespace`, `call_id` (I) | JSON rule below | `test_fr_gate_02_responses_function_call` |
| `function_call_output` | `output` string or parts (`input_text`, `input_image`, `input_file`) (R) | In place | `test_fr_gate_02_responses_function_call_output` |
| `custom_tool_call` | `input` (R, free text) | In place | `test_fr_gate_02_responses_custom_tool_call` |
| `custom_tool_call_output` | `output` string or parts (R) | In place | `test_fr_gate_02_responses_custom_tool_call_output` |
| `reasoning` | `summary[].text`, `content[].text` (S). `encrypted_content` (O) | A finding blocks (conservative, open question 3) | `test_fr_gate_02_responses_reasoning` |
| `local_shell_call` | `action.command[]`, `action.env` values, `working_directory` (R) | In place | `test_fr_gate_02_responses_local_shell_call` |
| `local_shell_call_output` | `output` (R, JSON string) | JSON rule | `test_fr_gate_02_responses_local_shell_call_output` |
| `shell_call` | `action.commands[]` (R) | In place | `test_fr_gate_02_responses_shell_call` |
| `shell_call_output` | `output[].stdout`, `output[].stderr` (R) | In place | `test_fr_gate_02_responses_shell_call_output` |
| `apply_patch_call` | `operation.diff`, `operation.path` (R) | In place | `test_fr_gate_02_responses_apply_patch_call` |
| `apply_patch_call_output` | `output` (R) | In place | `test_fr_gate_02_responses_apply_patch_call_output` |
| `computer_call` | `action` strings (typed text, URLs) (R) | In place | `test_fr_gate_02_responses_computer_call` |
| `computer_call_output` | `output.image_url` (M), `file_id` (I) | Pass with a warning | `test_fr_gate_02_responses_computer_call_output` |
| `web_search_call` | `action.queries`, `query`, `url`, `pattern` (R) | In place | `test_fr_gate_02_responses_web_search_call` |
| `file_search_call` | `queries` (R), `results[].text`, `filename` (R) | In place | `test_fr_gate_02_responses_file_search_call` |
| `code_interpreter_call` | `code`, `outputs[].logs` (R) | In place | `test_fr_gate_02_responses_code_interpreter_call` |
| `image_generation_call` | `revised_prompt` (R). `result` (M) | Pass the image with a warning | `test_fr_gate_02_responses_image_generation_call` |
| `mcp_call` | `arguments` (R, JSON string), `output` (R), `error` (R) | JSON rule | `test_fr_gate_02_responses_mcp_call` |
| `mcp_list_tools` | `tools[].description`, `input_schema` strings (R) | In place | `test_fr_gate_02_responses_mcp_list_tools` |
| `mcp_approval_request` | `arguments` (R, JSON string) | JSON rule | `test_fr_gate_02_responses_mcp_approval_request` |
| `mcp_approval_response` | `reason` (R) | In place | `test_fr_gate_02_responses_mcp_approval_response` |
| `tool_search_call`, `tool_search_output` | `arguments` (R, JSON), `tools` (as `tools[]`) | JSON rule, then as `tools[]` | `test_fr_gate_02_responses_tool_search` |
| `additional_tools` | `tools` (as `tools[]`) | As `tools[]` | `test_fr_gate_02_responses_additional_tools` |
| `item_reference` | `id` (I) | Pass. History the gate did not see: see the previous section | `test_fr_gate_02_responses_item_reference` |
| `compaction` | `encrypted_content` (O) | Pass. The summary is ciphertext the gate cannot read (residual risk, open question 2) | `test_fr_gate_02_responses_compaction` |
| any other `type` | n/a | **Block** (FR-GATE-03) | `test_fr_gate_02_responses_unknown_item_blocked` |

`tools[]` entries (all VERIFIED, O1). Text lives in these fields:

| Tool type | Scan (class) | Test |
| --- | --- | --- |
| `function` | `description` (R), `parameters` and `output_schema` string leaves (R). `name` (I) | `test_fr_gate_02_responses_tools_function` |
| `custom` | `description` (R), `format.definition` grammar (R) | `test_fr_gate_02_responses_tools_custom` |
| `mcp` | `server_label`, `server_description`, `server_url`, `allowed_tools` (R). `authorization`, `headers` (K) | `test_fr_gate_02_responses_tools_mcp` |
| `namespace` | `description` (R), nested `tools` (recurse) | `test_fr_gate_02_responses_tools_namespace` |
| `web_search`, `file_search`, `code_interpreter`, `computer*`, `image_generation`, `shell`, `local_shell`, `apply_patch`, `tool_search` | string settings such as `user_location`, `vector_store_ids` (R) | `test_fr_gate_02_responses_tools_builtin` |
| unknown `type` | **Block** | `test_fr_gate_02_responses_tools_unknown_type_blocked` |

## Rewrite rules

**1. Splice, never re-serialize.** The body is parsed into a tree that keeps the byte span of every token. The output is the input bytes with only the string literals that had findings replaced. Key order, whitespace, number formatting and escapes elsewhere stay as the client wrote them. A request with no findings is forwarded as the same bytes.

**2. A leaf is one decoded JSON string.** The scanner sees the decoded text, so `AKIA...` is scanned as `AKIA...`. A leaf with findings is replaced by a freshly encoded literal (escape `"`, `\` and control characters only). The placeholder contains none of those, so JSON validity cannot break.

**3. JSON inside a string** (`function_call.arguments`, `mcp_call.arguments`, `local_shell_call_output.output`, `tool_search_call.arguments`).

*Decision: parse, redact string leaves, splice back.* Do not redact the raw string. A raw-string span can start or end inside an escape sequence (`\n`, `\"`, `é`) and leave a dangling backslash, which breaks `arguments` and makes the tool call fail. Rule:
- If the string parses as JSON, scan its string leaves and splice replacements into the raw string, then encode the whole raw string once as the outer literal. A clean one is forwarded unchanged.
- If a leaf is itself a JSON object or array, recurse, to a depth of 3. Deeper is scanned as plain text.
- If the string does not parse (models do produce bad JSON), scan it as plain text and replace spans.
- Guard: a string that parsed before redaction must still parse after. If not, block.
- Free-form object keys are scanned. A finding in a key blocks, because two keys could collapse to one placeholder.
- A secret split across two leaves is not found. Residual, same class as FR-CORE-04.

**4. Anthropic `tool_use.input` and `server_tool_use.input` are objects, not strings.** Same leaf rule, no outer encoding step.

**5. Scanner context.** Each leaf is scanned with its key name as a hint (`password`, `token`) so name features still work. The hint is part of the pure function in the next section.

## Determinism and caching (FR-GATE-05)

Prompt caching and thinking binding both need identical prefix bytes across turns. Redaction of a leaf must therefore be a pure function of: the leaf text, its key hint, the session key (SEC-02) and the config epoch (thresholds, model version, allowlist). It must not depend on other leaves, later messages or earlier turns.

- No retroactive propagation. If a secret is found in turn 5, turn 1 text that held the same value but was not detected stays as it was sent, or the cache and thinking break (open question 5).
- Config changes mid-session (a threshold edit, a new allowlist entry under FR-GATE-08) change old bytes. Count as a cache break and log the epoch change.
- A gate restart drops the in-memory key, so placeholders change and every prefix differs. Expect a cache miss, and a thinking 400 that Claude Code recovers from (C1). Open question 6.
- `cache_control`, block order, block count and message count are never touched.
- Test: `test_fr_gate_02_determinism_prefix_identical_across_turns` replays a 5-turn history and compares prefix bytes. `test_fr_gate_02_determinism_cache_control_untouched` compares every `cache_control` byte.

## Request-level invariants

| Invariant | Behaviour | Test |
| --- | --- | --- |
| Clean body is byte-identical | Output equals input when nothing is found | `test_fr_gate_02_bytes_identical_when_clean` |
| Duplicate keys in any object | **Block.** Two parsers could disagree on which wins | `test_fr_gate_02_duplicate_key_blocked` |
| Invalid UTF-8, invalid JSON, nesting over 128, a non-object top level | **Block** | `test_fr_gate_02_malformed_blocked` |
| Max body size | Configurable, proposed default 64 MiB. Over it: block (FR-CORE-05, open question 7). The cap applies after decompression | `test_fr_gate_02_max_body_blocked` |
| Buffering | v1 buffers the whole body, bounded by the cap. The gate cannot forward any byte until it knows no block applies, because a partial send would leak. Leaves are scanned in chunks through the core streaming API | `test_fr_core_05_gate_100mb_memory_cap` |
| `Content-Encoding` | Absent or `identity`: pass. `gzip`, `zstd`: decode with the cap enforced on output size (a bomb is a block), forward uncompressed with the header and `Content-Length` fixed. Anything else (`br`, `deflate`, a list, unknown): **block** (FR-INSTALL-04) | `test_fr_gate_02_content_encoding_gzip`, `..._zstd`, `..._unknown_blocked`, `..._bomb_blocked` |
| Chunked request body | Supported (no `Content-Length`) | `test_fr_gate_02_chunked_body` |
| Scanner error or timeout | **Block** (FR-GATE-03) | `test_fr_gate_03_*` (FR-GATE-03 owns these) |
| Block response | Provider-shaped error, HTTP 400, with a reason code. The text must avoid the phrases Claude Code recovers from: `thinking`, `signature`, `bound to a different conversation`, `Extra inputs are not permitted`, `prompt is too long` (C1). Open question 8 | `test_fr_gate_02_block_message_has_no_recovery_phrases` |
| Images and PDFs (FR-GATE-07) | Pass with a warning in the audit log (type, size, no content) | `test_fr_gate_07_media_pass_with_warning` |

Audit entries (FR-LOG-01) name the block type and field path (`messages[7].content[2].input.cmd`), never the value.

## Test plan

- One parser test per row above, named as listed. Each plants a generated canary (never a real credential) in the named field, runs the parser, and asserts: the canary is gone, the placeholder is there, and every other byte is unchanged.
- Each S-class row has a second test: the canary blocks, with the FR-GATE-10 message.
- Each M-class row asserts the warning entry and the untouched bytes.
- Cross-cutting tests are in the invariants table. Add a fuzz target for the tree parser under SEC-04 in M3.
- The corpus for these tests is `bench/gate_corpus.py` (G2): one synthetic request per row above, in both formats, each with a generated canary in the named field, and each case carries the test name from the row. Cases whose block, item or tool type an agent actually sent in the G2 captures are marked `source: captured`; the others are written from the tables and the official docs (V6).

## Open questions for Josh (decided, see Decisions at the top)

1. **`previous_response_id` policy.** Block unless the gate relayed that id (proposed), or pass with a warning? Proposed is safer and costs a stream tee.
2. **OpenAI `compaction` items and Anthropic `encrypted_*` fields are ciphertext.** The gate cannot read what the provider already put there. Accept as a documented residual risk (proposed)?
3. **S-class conservative blocks.** `reasoning.summary`, `web_search_result.title/url` and `cited_text` might accept rewriting. Whether the provider validates them against the ciphertext needs a live API call with a fake secret. Not testable on the mock server. Until checked, a finding blocks. Do you want to run the live check, or keep blocking?
4. **K-class credentials** (`mcp_servers[].authorization_token` and the Responses `mcp` tool `authorization`): pass with a warning (proposed), or block? Claude Code and Codex do not use them by default.
5. **Cross-occurrence propagation.** Redacting a value everywhere once seen improves recall and breaks cache and thinking for old turns. Proposed: off in v1.
6. **Persist the session key across gate restarts?** SEC-02 says in-memory. Persisting keeps the cache and thinking valid across a restart but stores a key on disk.
7. **Body cap.** 64 MiB proposed. FR-CORE-05 scans 100 MB inputs in the core; the gate cap is a separate, lower limit. Is 64 MiB right?
8. **Block status code.** HTTP 400 proposed. Confirm how each agent shows it in G2. Answered in G2: both agents show the message and exit with status 1; Claude Code sends the request once more (`docs/gate-transport.md`, T3).
9. **Thinking-block findings.** The requirement says block and tell the user to start a new session. The API also allows removing thinking blocks entirely (A2, VERIFIED), which would keep the session alive. Do you want that as an option?

## To verify

G2 results, Oct 19, 2026. Probe run `20261019T175710Z-g2` (`bench/results/proof/g2-probes-20261019T175710Z-g2.json`: Claude Code 2.1.303, Codex CLI 0.163.0, fake keys, a mock provider on 127.0.0.1, a throwaway HOME per session). The baseline sessions are run `20261019T175822Z-agent-read-hook-s0`. The report holds header and field names, shapes and counts, never a value; the recordings (request bodies) stay in the ignored `bench/results/mock/<run>/`. Totals behind V3, V5 and V7: 615 model requests (Claude Code 21, Codex 594) in the probe run, the baseline run and the control run `20261019T175822Z-agent-control-s0`.

| # | Claim | Result |
| --- | --- | --- |
| V1 | Exact leading text of Claude Code's attribution system block, so the scanner can leave it intact and not mistake its fingerprint for a secret | **VERIFIED** (probe `claude-corpus`). `system` is a list of three `text` blocks. Block 0 is the attribution block alone, with no `cache_control`. Exact leading text: `x-anthropic-billing-header: cc_version=`. Full shape: `x-anthropic-billing-header: cc_version=<version>.<hex3>; cc_entrypoint=<entrypoint>;`, here `cc_version=2.1.303.850; cc_entrypoint=sdk-cli;` (`claude -p`). The 3 hex characters were `850` in every request of the six sessions that used the same prompt and `e03` in the session with a 700 KB prompt (an earlier ad-hoc run with the prompt "say hi" gave `3a9`), so they look derived from the prompt and are stable within a session. Blocks 1 and 2 carry `cache_control`. It is not a secret and the scanner can keep it as its own first block |
| V2 | The API accepts rewritten readable text in provider-produced blocks | **STILL OPEN, Josh's.** Needs a live API call with a fake secret; not reachable with a mock |
| V3 | Claude Code and Codex send no `Content-Encoding` on the Messages path; Codex over a custom provider sends none | **VERIFIED** (probes `claude-corpus`, `claude-long`, `codex-corpus`, `codex-long`, `codex-chatgpt-route-a`; baseline). No `Content-Encoding` on any of Claude Code's 21 requests (largest 764,520 bytes) or on Codex route A requests with an API key (largest 1,039,744 bytes) or a fake ChatGPT login. The one request with an encoding was Codex on route B with a fake ChatGPT login: `zstd` (T7) |
| V4 | Claude Code resends `thinking` and `redacted_thinking` blocks | **VERIFIED, client side** (probe `claude-corpus`). The mock replied with a `thinking` block and a `redacted_thinking` block; request 1 carried the `thinking` block and requests 2 to 4 carried both blocks in the assistant message, with the signature and the data string unchanged. Codex resends `reasoning` items with their `encrypted_content` the same way (probe `codex-corpus`). Limit: the blocks were the mock's, so this does not show that the API accepts an edited one (V2) |
| V5 | Codex HTTP bodies carry no `previous_response_id` | **VERIFIED**. None in 592 route-A request bodies (API key and fake ChatGPT login; probes and baseline) or in the 2 HTTP fallback bodies of route B. No `conversation` either. All 592 route-A bodies have `store: false`, `stream: true`, `include: ["reasoning.encrypted_content"]`, an `input` list holding the whole conversation, and no `instructions` |
| V6 | Which block and item types the agents actually emit, so the test corpus is real | **VERIFIED for what was sent; the rest is still spec-only.** Claude Code sent content blocks `text`, `thinking`, `redacted_thinking`, `tool_use`, `tool_result` (content a string, or blocks holding an `image`), `role: "system"` messages (see G2 findings), `tools[]` of the custom form only, and top-level `model`, `max_tokens`, `messages`, `system`, `tools`, `metadata`, `thinking`, `context_management`, `output_config`, `stream`. Codex sent `input` items `message` (roles `developer`, `user`, `assistant`; parts `input_text`, `output_text`), `reasoning`, `function_call`, `function_call_output`; tools `function`, `namespace` (the Harpo MCP tools sit inside one), `web_search`; and top-level `model`, `input`, `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning`, `store`, `stream`, `include`, `prompt_cache_key`, `client_metadata` (no `instructions`). No agent sent any other type in the scripted sessions (documents, search results, server-tool results, MCP connector blocks, compaction blocks, shell and patch items), so those rows stay as written from the docs. `bench/gate_corpus.py` marks the 20 captured cases `source: captured` |
| V7 | Request bodies arrive chunked or with `Content-Length` | **VERIFIED**. All 615 model requests carried `Content-Length`; none was chunked, up to 1,039,744 bytes. The gate still supports chunked bodies (the spec requires it; the mock handles both) |

### G2 findings that touch the tables above

These came out of the captures. None changes a decision; each needs a line in a table or a test when G4 is written, so they are listed for Josh.

1. **Mid-conversation `system` messages with string content.** Claude Code 2.1.303 puts `role: "system"` entries in `messages[]`, in two forms: a list holding one `text` block (the newest one, carrying the moving `cache_control` marker) and a plain string (older ones: an 8.5 KB "# Environment ..." message with cwd and OS details, and short `<total_tokens>` reminders). The same message is a list in one request and a string in the next. The table above lists the block form only. A string-content `system` message must be scanned and rewritten in place, and kept a string (rule: never convert string and array). Corpus case: `anthropic_system_role_string`, no spec test name yet.
2. **`x-codex-turn-metadata` is in two places.** Codex sends it as a header and again as `client_metadata["x-codex-turn-metadata"]` in the body (the same JSON string), with `session_id`, `thread_id`, `turn_id`, `x-codex-window-id` and `x-codex-installation-id` as further `client_metadata` entries. See T8 in `docs/gate-transport.md`: it holds an absolute workspace path and a 40-hex commit hash, which a hex or entropy scanner will meet on every request.
3. **Codex has no `instructions` field.** Its system prompt is two `developer`-role `message` items at the start of `input`.
4. **Codex MCP tools are a `namespace` tool.** The Harpo server shows up as `{"type": "namespace", "name": "mcp__harpo", "tools": [function, ...]}` in `tools[]`, which the `namespace` row already covers.
5. **Claude Code's first user message has four `text` blocks**, and `metadata.user_id` is a JSON string (`device_id` as 64 hex characters, `account_uuid`, `session_id`). Both are in the tables as plain `R` text; the second is JSON inside a string, and its `device_id` is a hex string a scanner may flag.
6. **One retry on any 400** (`docs/gate-transport.md`, T3): Claude Code sends the same request again without `thinking.display` and without one `anthropic-beta` value. A gate block is therefore seen twice.

## Sources (all checked 2026-10-19)

IDs match the sources table in [gate-transport.md](gate-transport.md); CX7 is used only here.

| ID | URL |
| --- | --- |
| A1 | https://platform.claude.com/docs/en/api/messages |
| A2 | https://platform.claude.com/docs/en/build-with-claude/preserved-thinking |
| A3 | https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages |
| A4 | https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand |
| A5 | https://platform.claude.com/docs/en/agents-and-tools/mcp-connector |
| A6 | https://platform.claude.com/docs/en/api/messages/count_tokens |
| C1 | https://code.claude.com/docs/en/llm-gateway-protocol |
| O1 | https://developers.openai.com/api/reference/resources/responses/methods/create |
| O2 | https://developers.openai.com/api/docs/guides/websocket-mode.md |
| CX1 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/core/src/client.rs |
| CX7 | https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/codex-api/src/common.rs |
