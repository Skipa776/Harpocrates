# Spec: secrets base64-encoded inside text fields

Oct 20, 2026 · status: **approved by Josh, Oct 20, 2026** · requirement: **FR-GATE-11 (new, M2)**. Builds on `docs/gate-scan-spec.md` (G4) and `docs/gate-media-upstream-spec.md` (G5, "End-to-end result for media").

## Why

G5's end-to-end run found an agent path the media rule doesn't cover. Codex sends an MCP embedded resource as JSON *text*, with the file's base64 blob inside an `input_text` part. The gate scans that text as text and never decodes it.

The gate-on run received 0 canaries, but only because the detector flagged the whole blob as a `generic` finding. The cause is entropy, not intent:
- a short or less random blob can pass;
- a harmless blob, such as a real PDF, gets redacted.

The same gap covers any base64 the model sees in text: a tool result that prints `base64 .env`, a Kubernetes Secret manifest, or a `data:` URL pasted into a prompt.

## Rule

For every class R and S string the walkers yield, after the existing text scan, and including the string leaves of JSON-in-string values:

1. **Find runs.** A run is 24 or more characters from the base64 alphabet `[A-Za-z0-9+/]`, or the base64url alphabet `[A-Za-z0-9_-]`, with optional `=` padding. Runs are found on the decoded JSON string, so escapes like `\n` don't break or shift them. A run broken by line breaks (MIME wrapping at 64 or 76 characters) is joined first.
2. **Decode at every alignment.** Try offsets 0 to 3, padding a partial last group, so a run glued to a stray letter still decodes. The harness's `received_in_base64` showed this matters.
3. **Keep only text.** A decoding counts only when it is valid UTF-8 and at least 90% printable, not counting tabs and newlines. Anything else is binary or garbage and isn't scanned. This keeps images, compressed data, hashes and random IDs out of the detector.
4. **Scan.** Each kept decoding goes through the same detector at the gate threshold. Results are cached by SHA-256 of the run, under a new cache class `B`.
5. **Block on a finding** with a new reason, `HARPO-BLOCK-ENCODED-SECRET`. The message names the code only, then says: "Text in this request holds base64-encoded data with a secret in it. Remove it or send the decoded text, so Harpo can redact it."
6. **One level only.** A decoding is not searched for further base64.

Placeholders (`<<HARPO:...>>`) and media fields (class M, G5) are not affected. Every error in this step blocks, as G4 already does for detector errors.

## Trade-offs (rule 4)

| Choice | Gain | Cost |
| --- | --- | --- |
| **Block, not redact** | Simple and fails closed. Matches G5's choice for base64 text files | One encoded secret stops the whole request. Today the entropy redaction lets the session go on. See open question 1 |
| **Minimum run of 24 characters** (18 bytes) | Catches a secret of about 18 bytes or longer, encoded on its own | A shorter encoded secret is only seen as raw text. Lowering the floor adds decodes but few findings |
| **Text filter (UTF-8, 90% printable)** | Binary, compressed and random runs never reach the detector, so no new false positives come from them | A secret inside encoded binary, such as a PDF's text stream or gzip then base64, still passes. That is the same accepted limit as G5 |
| **All four alignments** | Glued or escaped runs are still found | Up to 4 decodes per run. Garbage alignments fail the text filter |
| **One decoding level** | Bounded work | Double-encoded secrets pass |
| **Scanning decoded text at the gate threshold** | Same recall as plain text | Benign encoded text that trips the detector blocks the request. Examples: a source map in a `sourceMappingURL` data URL, an encoded config with a high-entropy ID. The G4/G5 runs already showed entropy false positives on paths and identifiers (M2 plan, D4). Here each one becomes a block, not a redaction |

**Performance (NFR-01).**
- Finding runs is a linear scan of text the gate already scans.
- Decoding and scanning happen only for runs not seen before (cache class `B`), so follow-up turns rescan nothing old.
- A very large encoded blob is bounded by the 5 s scan timeout, which blocks.
- The G2 replay latency is re-measured and published in `bench/results/gate/` with the change.

## Tests, first

- `test_fr_gate_11_encoded_secret_in_text_blocks`, for each format:
  - an Anthropic `text` block and a `tool_result` string;
  - Responses `input_text`, `function_call_output` and `function_call.arguments` (JSON in a string).
- `test_fr_gate_11_codex_mcp_resource_blob_blocks`: the exact G5 shape, `{"type":"resource",...,"blob":"<b64>"}` as `input_text`.
- `test_fr_gate_11_misaligned_wrapped_and_base64url_runs_are_decoded`: a run after `x\n`, after a stray letter, MIME-wrapped at 76 characters, and URL-safe.
- `test_fr_gate_11_binary_and_random_runs_are_not_scanned`: a PNG, gzip output and a SHA-256 in base64 pass unchanged.
- `test_fr_gate_11_short_run_is_not_decoded`: 23 characters.
- `test_fr_gate_11_clean_encoded_text_passes_byte_identical`
- `test_fr_gate_11_decoder_error_blocks`
- `test_fr_gate_11_cache_holds_no_values` (SEC-02)
- **End to end (rule 7):** rerun suite `g5`. `codex-mcp-gate` must now end with `HARPO-BLOCK-ENCODED-SECRET` in the gate log, and the mock must record no request after it. Every other gate-on scenario must be unchanged.
- **False-positive check:** run the rule over the recorded G2, G4 and G5 sessions and report how many requests it would block. The target is 0 outside the canary cases. If it is above 0, bring it back to Josh before tuning anything (rule 5: those recordings are not used for tuning).

## Not covered

- hex, URL-encoded or other encodings
- gzip and other compression under base64
- double encoding
- a run split across two fields
- secrets inside encoded binary

Each one is a separate decision if needed.

## Implementation notes (Oct 20, 2026, after review)

The code is `cli/Harpocrates/gate/encoded.py` (run finder, decoder, text filter). `RequestScanner._encoded` runs it on every class R and S leaf, after that leaf's own scan.

The security and python reviews found bypasses of rules 1 and 3 as first written. All of these are fixed, so the rule is now stricter than the text above. The tests are in `cli/tests/test_fr_gate_11_encoded_text.py`.

- **Rule 1, separators.** A run may be split by spaces, tabs, real line breaks, or the two-character escapes `\n` and `\r\n` left in text, as when a tool result prints a JSON or notebook file. The split joins only between pieces of 16 or more characters, plus a short last piece, so prose never becomes a run.
- **Rule 3, text filter.**
  - A decoding is scanned whole when at least 90% of it is printable. Otherwise only its printable stretches of 32 or more characters are scanned, which catches a secret next to stray bytes, Latin-1 accents or a long garbage prefix. Random and compressed bytes essentially never hold a 32-character printable stretch.
  - The bytes are also read as UTF-16, little- or big-endian, when their NUL bytes fall every other byte. A `.env` written by PowerShell then base64-encoded is caught.
  - Bad UTF-8 bytes are replaced and count as not printable.
- **Size cap (new).** One request may decode at most 2 MiB of base64 run characters. The budget is checked as runs are found, before any decoding. Past it the request blocks at once with `HARPO-BLOCK-ENCODED-TOO-LARGE`, not after a 5 s timeout. That leaves no scan thread decoding after its request was already blocked.
  - **Why 2 MiB:** it sits below what one 5 s scan can decode (about 3 MB, measured in review) and far above anything recorded. The largest recorded request holds 2,202 run characters.
  - **Trade-off (rule 4):** a legitimate request with more than 2 MiB of base64 in text is blocked. It would have hit the scan timeout and been blocked anyway, only more slowly.
  - Runs in leaves already found clean on an earlier turn don't count against the cap.
- **Cache.**
  - A leaf found clean is remembered by SHA-256 under class `E`, or `J` when it was read as JSON. Keeping them apart matters because a JSON reading is searched more widely (escapes decoded). Text first seen as plain text, then sent as tool-call arguments, is searched again.
  - Each decoding's findings are cached under class `B`. No text is kept (SEC-02).
- **Order in a class S field.** The raw text is scanned first. If the detector flags the run itself, the request blocks with `PROVIDER-TEXT` before decoding.
- **Still accepted limits** (each needs deliberate evasion rather than an accident, per the second security review):
  - a secret interleaved with a non-printable byte every few bytes, which leaves no 32-character stretch;
  - a secret under 32 characters inside a mostly binary decoding;
  - a run split into pieces under 16 characters, or by commas, `+` or zero-width characters;
  - encodings other than base64;
  - compression;
  - double encoding.
- **Follow-up, not FR-GATE-11's to fix.** A scan that times out is blocked, but its worker thread cannot be cancelled and runs on. That is how the scan step has worked since G4. The size cap keeps base64 decoding from adding to it, but the general fix (a deadline checked inside the scan, or a process pool) belongs to its own requirement.

## Results (Oct 20, 2026, final code)

- **False-positive check.** `python -m bench.gate_encoded_fp`, report `bench/results/gate/encoded-fp-2026-10-20.json`, covering every recording in `bench/results/mock` (G2, G3 retro, G4, G5 and the canary harness runs).
  - Volume: 1,038 unique requests on scan routes, 26,024 base64 runs, and 13 decodings with text.
  - **Would block: 3 requests.** All are the Codex MCP gate-off controls (`g5f`, `g5g`, `g5h`), whose blob is the canary `.env`. **0 outside the canary cases.**
- **End to end.** Suite `g5`, report `bench/results/proof/20261020-gate-e2e-g5i.json`.
  - `codex-mcp-gate` ends with `HARPO-BLOCK-ENCODED-SECRET`. Codex shows the gate's message, the mock records 3 requests (the blocked fourth never left), and 0 canaries are received.
  - The other seven scenarios are unchanged. The PNG stays byte-identical, and the Codex MCP control still leaks all 5 canaries.
- **Latency (NFR-01).** `python -m bench.gate_scan_latency` on the G2 runs, report `bench/results/gate/scan-latency-2026-10-20-fr-gate-11.json`. G4's file is left as published.

  | Case | G4 p95 | p95 with FR-GATE-11 |
  | --- | --- | --- |
  | Follow-up turns | 8.8 ms | 12.2 ms |
  | Large follow-ups | 11.8 ms | 14.2 ms |
  | First turns | 318 ms | 347 ms |

  The follow-up target (under 50 ms) is still met.

## Decisions

Josh approved the spec as written on Oct 20, 2026.

1. **A finding blocks.** The gate does not redact the whole run.
2. **FR-GATE-11 is in M2.** It needs a row in the requirements table of `docs/design.md`, added with the code.
