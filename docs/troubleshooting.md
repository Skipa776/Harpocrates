# Troubleshooting

Each entry gives the symptom, the cause and the fix. Entries were checked on Oct 21, 2026 unless they say otherwise.

Never paste a real secret into an issue or a log. Harpo's messages and logs name secret types and paths, never values.

For setup, see [setup.md](setup.md). For running the tests, see [testing.md](testing.md).

## Install and environment

**The model will not load, or Harpo refuses a model artifact.**

Cause: In a clone, the model files come from Git LFS. Without `git lfs pull`, `cli/Harpocrates/ml/models/model.onnx` is a pointer text file of about 130 bytes. Harpo also refuses any artifact whose SHA-256 does not match `onnx_model_hashes.json`, and never loads it (SEC-07).

Fix: Run `git lfs install`, then `git lfs pull`. `file cli/Harpocrates/ml/models/model.onnx` should no longer say "ASCII text". The PyPI package ships the real files.

**`ruff check .` gives a different result from CI.**

Cause: Another ruff is first on PATH.

Fix: Compare `ruff --version` with `python -m ruff --version`, and use `python -m ruff check .`.

**The old command name prints a deprecation line.**

Cause: The commands were renamed (FR-NAME-02). The rename note near the end of the README lists every change.

Fix: Use `harpo` and `harpo-mcp`. The old names still work for two minor releases.

## Tests

**pytest stops at collection with `No module named 'zstandard'` or `No module named 'yaml'`.**

Cause: Before Oct 21, 2026, the dev extra did not declare these two test dependencies, so a clean install could not collect the suite.

Fix: Re-run `pip install -e ".[dev,explain,mcp]"`.

**About 70 CLI tests fail with `AttributeError: 'CliRunner' object has no attribute 'catch_exceptions'`.**

Cause: typer 0.27 made its CliRunner a separate class rather than a click subclass. The test helper `cli/tests/cli_entry.py` called click's method on it. Fixed on Oct 21, 2026.

Fix: Update the checkout.

**`test_fr_name_01_generated_codex_config_uses_the_harpo_names_and_keeps_every_deny` fails when the virtual environment sits inside a clone of this repository.**

Cause: The generated Codex config holds the interpreter's path, and the test checked the whole text for the project's old name, which a clone's folder carries. Fixed on Oct 21, 2026.

Fix: Update the checkout. The test now ignores the interpreter path.

**6 tests skip.**

Cause: Expected. 5 skip when run as root, because root ignores file modes. 1 needs a case-insensitive file system, which is the macOS default.

Fix: None needed. Any other skip needs a reason.

**An end-to-end harness exits 2.**

Cause: The result is inconclusive: an agent CLI is missing or did not finish.

Fix: Install `claude` and `codex` with `npm install -g @anthropic-ai/claude-code @openai/codex`. Never log them in. Run again. An exit code of 0 from `python -m bench.gate_e2e` only means the real home did not change. Read each scenario in its report, as described in [testing.md](testing.md).

## The read hook

**A Read, Grep or `cat` of a file is refused with a pointer to `safe_read`.**

Cause: The file holds something the detector flags. Reads never prompt (FR-READ-06).

Fix: Let the agent use `safe_read` or `safe_grep`, which return the file with placeholders. If the file is a false alarm and must be read as it is, allow it yourself in a terminal with `harpo allow PATH`. Every use is logged to `~/.harpo/allow.log`. See [setup.md](setup.md#allowlisting-paths-and-names).

**The agent's own `harpo allow` command, or its edit of `~/.harpo`, is refused.**

Cause: This is by design. Only you can change the allowlist (FR-READ-05, SEC-09).

Fix: Run the command yourself in a terminal.

**In Codex, the Harpo hook never runs.**

Cause: Codex runs a new or changed hook only after you trust it.

Fix: Start `codex`, type `/hooks`, and trust the Harpo hook.

## The egress gate

`harpo guard up` starts the gate and routes the agents to it; `harpo doctor` checks the setup. The gate's request log is ~/.harpo/gate.log: one JSON line per request, with the decision and reason code, never a header value or body.

**A block message says "Run `harpo doctor` for details."**

Cause: The gate refused the request; the reason code in the message says why (table below).

Fix: Run `harpo doctor`: it checks the gate and the routing and names the fix for each failure.

**Every request from Codex with a ChatGPT login is blocked with HARPO-BLOCK-ENCODING.**

Cause: Those requests are zstd-compressed. On Python before 3.14 the gate decodes them with the `zstandard` package, a dependency of Harpo since Oct 21, 2026; an older install, or one made with `--no-deps`, may lack it, and the gate then blocks rather than forward a body it cannot scan.

Fix: Reinstall Harpo, or run `pip install zstandard` in the gate's environment.

**One prompt shows the same block several times, or the request log has four blocked requests for one prompt.**

Cause: Claude Code retries a request that was answered with 400. Version 2.1.306 sent it four times in total (2.1.303: twice). Every retry is blocked too.

Fix: None needed.

**Every tool call in a Claude Code session is refused with a message containing "Harpo gate bypass".**

Cause: SEC-10. The install record says the gate is on, but this session's settings route Claude Code elsewhere. The override is one of these: an `ANTHROPIC_BASE_URL` in the environment, or in `env` of `.claude/settings.json`, `.claude/settings.local.json` or `~/.claude/settings.json`, that is not the gate; `CLAUDE_CODE_USE_BEDROCK=1`; `CLAUDE_CODE_USE_VERTEX=1`; or `ANTHROPIC_CUSTOM_HEADERS` without the gate token. The refusal names the setting file.

Fix: Remove the override from that file or the environment.

**On Python 3.13, before Oct 21, 2026, an HTTP/0.9 request is blocked with HARPO-BLOCK-INTERNAL instead of HARPO-BLOCK-TOKEN.**

Cause: Python 3.13 parses HTTP/0.9 headers differently. The request was blocked either way.

Fix: Update the checkout. This was fixed on Oct 21, 2026.

**`harpo guard up` refuses Claude Code or Codex.**

Cause: That harness already routes elsewhere. The settings it refuses on are its own `ANTHROPIC_BASE_URL` or `ANTHROPIC_CUSTOM_HEADERS`, a `CLAUDE_CODE_USE_<provider>` setting that is on, or, for Codex, its own `model_provider` or `openai_base_url`. Chaining to an existing gateway is not supported yet.

Fix: Remove that setting. Or leave that harness off the gate by naming only the other one with `--claude-code` or `--codex`.

**Requests fail to connect after a reboot or after the gate stopped.**

Cause: The agents are still routed to a gate that is not running. Nothing is sent unscanned; the requests just fail.

Fix: Run `harpo guard up` to start the gate again, or `harpo guard down` to take the agents off the gate.

**Codex through the gate gets no Authorization header.**

Cause: With codex-cli 0.163.0, an API key that is only in the `OPENAI_API_KEY` environment variable was not sent through a custom provider (see [gate-transport.md](gate-transport.md), T4).

Fix: Run `codex login`, with a ChatGPT login or `codex login --with-api-key`.

## Block reason codes

| Code | HTTP status | Meaning |
|---|---|---|
| HARPO-BLOCK-INTERNAL | 400 | An unexpected error in the request pipeline, before any reply was sent. The error text is never shown or logged. |
| HARPO-BLOCK-ROUTE | 400 | A route or method the gate does not know; unknown routes are blocked, never forwarded (gate decision 13). |
| HARPO-BLOCK-UPGRADE | 426 | A WebSocket upgrade request; the gate refuses it with 426 and Codex falls back to HTTP (gate decision 3). |
| HARPO-BLOCK-UNKNOWN-TYPE | 400 | A block, item, tool or source type that the parser spec does not list. |
| HARPO-BLOCK-TOKEN | 401 | The gate token header check failed. |
| HARPO-BLOCK-SCAN-TIMEOUT | 400 | A scan step ran past its timeout, and a late answer is dropped. |
| HARPO-BLOCK-SCAN-ERROR | 400 | A scan step raised an exception, returned an unsafe block reason, or returned an answer of an unknown kind. |
| HARPO-BLOCK-REQUEST | 400 | A request that failed to parse, for a reason other than an unknown method. |
| HARPO-BLOCK-ENCODING | 400 | A Content-Encoding the gate does not support, or zstd without a zstd module. |
| HARPO-BLOCK-BODY-TOO-LARGE | 400 | A body, or a decoded body, over the size cap. |
| HARPO-BLOCK-PROVIDER-TEXT | 400 | A secret in text that only the provider can rewrite. |
| HARPO-BLOCK-MEDIA-SECRET | 400 | A secret in a text file sent as base64, which is not rewritten in v1. |
| HARPO-BLOCK-ENCODING-CORRUPT | 400 | A compressed body that is bad, truncated or padded. |
| HARPO-BLOCK-ENCODED-TOO-LARGE | 400 | More base64 in text than one request may decode. |
| HARPO-BLOCK-ENCODED-SECRET | 400 | A secret in base64 inside a text field. |
| HARPO-BLOCK-BODY-TRUNCATED | 400 | The sender stopped, timed out or hung up before the body ended. |
| HARPO-BLOCK-BODY-FRAMING | 400 | Invalid Content-Length or chunk framing, or both framings sent at once. |
| HARPO-BLOCK-PARSE | 400 | Text that is not UTF-8, not JSON, has a repeated key, is nested too deep, or is not an object. |
| HARPO-BLOCK-MEDIA-MALFORMED | 400 | A media field that is not strict base64 to its end. |
| HARPO-BLOCK-KEY-SECRET | 400 | A secret in an object key, where two keys could collapse into one placeholder. |
| HARPO-BLOCK-JSON-ARGS | 400 | Tool-call arguments that parsed before redaction but not after. |
| HARPO-BLOCK-HISTORY | 400 | Server-side history the gate cannot see, such as previous_response_id or a conversation. |
| HARPO-BLOCK-HEADER-SECRET | 400 | A secret in a header value, the path or the query, none of which can be rewritten. |
