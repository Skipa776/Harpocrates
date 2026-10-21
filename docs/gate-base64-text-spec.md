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

## Decisions

Josh approved the spec as written on Oct 20, 2026.

1. **A finding blocks.** The gate does not redact the whole run.
2. **FR-GATE-11 is in M2.** It needs a row in the requirements table of `docs/design.md`, added with the code.
