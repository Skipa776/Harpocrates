# G4 spec: scan and redact the request

Oct 19, 2026 · status: **awaiting Josh's approval** · requirements: FR-GATE-03, FR-GATE-04, FR-GATE-05 (determinism part), FR-GATE-10, FR-CORE-05, SEC-02. Builds on `docs/gate-parser-spec.md` (field classes, block tables and the 15 decisions) and replaces G3's blocking stub.

## Inputs and outputs

**Input:** the decoded body, route, request headers, path and query. G3's `(body, route)` signature widens to this.

**Output:** either `Forward(bytes)` or `Block(reason)`. The forwarded bytes equal the input bytes exactly, except inside rewritten string values.

## Steps

1. **Parse strictly (decision 15).** UTF-8 JSON, no duplicate keys, nesting at most 128 deep. Anything else → block.
2. **Walk the request** with the parser spec's per-format tables: every system field, message, content block, input item, tool definition and tool argument. Each value gets its field class:
   - **R** rewrite
   - **S** scan only
   - **I** ignore
   - **O** opaque, can't be rewritten
   - **M** metadata
   - **K** credential
   
   An unknown block or item type blocks the request.
3. **Drop thinking (decision 5, FR-GATE-10).** Remove every `thinking` and `redacted_thinking` block before scanning. For opaque content that can't be dropped (encrypted reasoning, provider-signed readable text): a finding → block.
4. **Scan.** Each R and S text value goes through the existing detector (`detect_text`) at the **gate threshold** from `model_config.json`. Gate threshold, not commit: recall first.
   - **JSON-in-string values** (tool `input`, `arguments`, decision 2): parse, scan the string leaves, splice the redactions back into the raw bytes. A secret in an object key blocks the request.
   - **Headers (decision 12):** every non-credential header value is scanned, and so are the path and query. A finding → block, because a header can't be safely rewritten.
5. **Redact.** Each finding's exact span is replaced with `<<HARPO:type:hash4>>`. `hash4` is the first 4 hex characters of HMAC-SHA256 under the **session key**: random, in memory only, made at gate start (SEC-02, decision 7). The same secret always gets the same placeholder within a gate session. Bytes outside rewritten spans never change.
6. **Splice.** The output is built by replacing byte spans in the original body, not by re-serializing JSON. This keeps key order, whitespace and number formatting byte-stable (FR-GATE-05).

## Performance: a per-session cache

The agent resends the whole history every turn, up to about 1 MB in the G2 captures. Scanning all of it each time can't meet NFR-01's 50 ms p95. Because redaction is deterministic, results are cached:

- **Key:** SHA-256 of `(field class, text value)`, so the cache never holds a raw value.
- **Value:** the list of findings as `(start, end, type)` spans.
- **Bounds:** at most 50,000 entries and 64 MiB, least-recently-used eviction, in memory only, cleared when the gate stops.

In practice each turn scans only the new turn's text. Earlier text is a cache hit, and the cache returns the same spans that produced the same placeholders. A cache miss rescans, which is slower but always correct.

**Target for this spec:** added p95 below 50 ms on G2-sized follow-up turns. Measured on replayed G2 recordings and published in `bench/results/gate/`. The first turn of a large session is reported separately.

## Failure behavior

Every failure blocks; none of these forwards anything:
- parse error
- unknown type
- opaque content with a finding
- secret in an object key or header
- detector exception
- scan timeout (default 5 s)
- body over the cap

**Override (FR-GATE-03):** `--allow-unscanned` exists only as a start-up flag of the gate process, never per request. It logs a loud line on every forwarded request, it is off by default, and `harpo doctor` reports it. **Your call on the exact wording, or whether to drop the override from v1.** My recommendation is to drop it.

## Test plan, tests first

- **`test_fr_gate_02_<format>_<type>`:** one per case of the G2 corpus (`bench/gate_corpus*.py`), 93 cases. Each asserts the canary is gone from the forwarded bytes, the placeholder is present and well formed, everything else is byte-identical, and blocked types are blocked.
- **`test_fr_gate_04_turn1_secret_redacted_in_turn5`**
- **`test_fr_gate_05_same_secret_same_placeholder_across_turns`** and **`test_fr_gate_05_clean_body_forwarded_byte_identical`**
- **`test_fr_gate_10_thinking_dropped_request_succeeds`** and **`test_fr_gate_10_secret_in_encrypted_reasoning_blocks`**
- **`test_fr_gate_03_detector_error_blocks`**, **`..._timeout_blocks`** and **`..._parse_error_blocks`**
- **`test_fr_gate_02_secret_in_json_key_blocks`** and **`test_fr_gate_02_secret_in_header_blocks`**
- **`test_sec_02_cache_holds_no_values`** and **`test_sec_02_new_gate_new_key`**
- **Canary harness against `gate → mock` (FR-PROOF-04):** `direct` mode reports 0 known-format canaries received. `agent` mode is run once G7 installs the gate.

## Not in G4

- images and PDFs (G5)
- the audit log (G6)
- install, `guard up` and `doctor` (G7)
- the ChatGPT-login upstream (`chatgpt.com/backend-api/codex`) and upstream connection pooling, both left open from G3, scheduled for G5 with your OK
