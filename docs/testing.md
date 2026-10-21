# Testing

This guide covers how to run Harpo's test suites and how to read their results. The end-to-end suites plant canaries (generated fake secrets in real provider formats) and count how many reach the provider through a protected path. A command that exits 0 is not proof. Open the report and check planted versus received, the number of tool calls made, and the skip list.

Tests are named after requirement IDs. A test for FR-GATE-03 is named `test_fr_gate_03_*`. A feature counts as done only when an end-to-end test exercises it through its real entry point: the `harpo` CLI, the hook as the agent calls it, or the gate with a real `claude -p` or `codex exec`.

## 1. Set up

Install Harpo with the dev, explain and mcp extras:

```bash
pip install -e ".[dev,explain,mcp]"
```

**Dev extra.** The dev extra includes pytest, pytest-cov, ruff, mypy, build, zstandard and pyyaml.

**Model files.** The model files are in Git LFS. Run `git lfs pull`. An unfetched `cli/Harpocrates/ml/models/model.onnx` is a text file of about 130 bytes. Harpo refuses any model artifact whose SHA-256 does not match `onnx_model_hashes.json` (SEC-07).

**Rust scanner.** For runs against the Rust scanner, build the release binary:

```bash
cargo build --release --manifest-path core/Cargo.toml
```

The package finds `core/target/release/harpocrates-rust-scanner` by itself.

**End-to-end CLIs.** The end-to-end suites need the `claude` and `codex` CLIs:

```bash
npm install -g @anthropic-ai/claude-code @openai/codex
```

Never log either CLI in. Every suite uses fake keys. When a CLI is missing, the harness reports the run as inconclusive (exit 2) and the integration test that drives the agents skips. Neither counts as a pass.

## 2. The checks CI runs

| Check | Command | Notes |
|---|---|---|
| Lint | `ruff check .` | Use the dev extra's ruff (`python -m ruff check .`) if another ruff is first on PATH. |
| Docstrings | `python scripts/check_docstrings.py --enforce` | Modules listed under `[tool.harpo.docstrings]` in pyproject.toml (DOC-DOCSTRING-01). |
| Unit tests and coverage | `pytest -q -m "not integration" --cov --cov-report=term` | Fails below `fail_under = 90` in pyproject.toml (TEST-01). |
| Unit tests on the Rust scanner | `HARPOCRATES_REQUIRE_RUST=1 pytest -q -m "not integration" --cov` | Run after the release build (FR-CORE-01). |
| Rust format | `cargo fmt --manifest-path core/Cargo.toml --check` | |
| Rust lint | `cargo clippy --manifest-path core/Cargo.toml --all-targets -- -D warnings` | |
| Rust tests | `cargo test --manifest-path core/Cargo.toml` | |
| Integration | `pytest -q -m integration` | Builds a wheel and runs real agents where installed. |
| Plugin | `claude plugin test plugins/harpocrates-prompt` | CI pins Claude Code 2.1.302 for this job. |

**Expected skips.** When run as root, 5 tests skip because root ignores file modes. One test needs a case-insensitive file system (the macOS default). Any other skip needs a reason.

**CI triggers.** `.github/workflows/ci.yml` runs on pushes to `main`, `feature/**` and `feat/**`, and on pull requests. A branch with another name gets no CI run until it has a pull request.

## 3. End-to-end suites

**Isolation.** Every agent runs in a throwaway HOME, CODEX_HOME, TMPDIR and repo, with an environment built from scratch, fake API keys, a loopback mock provider (`bench/mock_provider.py`) and, for gate suites, the real gate on 127.0.0.1. Proxy variables point at a dead port. The real `~/.harpo`, `~/.harpocrates`, `~/.claude/settings.json`, `~/.codex/config.toml` and `~/.codex/auth.json` are snapshotted before and after (`auth.json` by size and time only), and a report lists any change. Canaries are generated fakes in real provider formats (`bench/canary_repo.py`). The answer key never enters the repo under test.

**Output.** Pass `--out-dir DIR` to keep runs out of `bench/results/`. `DIR/proof/<run>.json` holds counts only, and the writer refuses a report that contains a canary value. `DIR/mock/<run>/` holds the recorded request bodies. It is gitignored and must never be committed.

| Suite | Command | Requirements | Passes when |
|---|---|---|---|
| Counter check | `python bench/canary_harness.py --mode direct --target mock` | FR-PROOF-04 | Every planted canary is received. This proves the counter works. |
| Parser corpus through the gate | `python bench/canary_harness.py --mode direct --target gate` | FR-GATE-02, FR-PROOF-04 | 0 known-format canaries received. Credentials for a provider-hosted MCP connector pass by design (gate decision 6) and are listed as `received_by_design`. ML-only canaries that get through are counted and published. |
| Read hook with real agents | `python bench/canary_harness.py --mode agent --agent all` | FR-READ-01, FR-READ-03, FR-READ-06, FR-PROOF-04 | Each agent made its 3 scripted tool calls, and 0 known-format canaries were received. |
| Read hook control | `python bench/canary_harness.py --mode agent --agent all --control` | Counter check | Known-format canaries are received, because nothing is installed. |
| Gate, fail closed and SEC-10 | `python -m bench.gate_e2e --suite g3` | FR-GATE-03, SEC-10, FR-GATE-01 | Both stub-scan sessions show the block, and 0 requests reach the mock. With a mismatched install record, every tool result carries the SEC-10 refusal. |
| Gate, scripted reads | `python -m bench.gate_e2e --suite g4` | FR-GATE-01, FR-GATE-04, FR-GATE-05 | Gated sessions receive 0 canaries, and their tool results carry placeholders. The gate-off controls leak. |
| Gate, media and MCP images | `python -m bench.gate_e2e --suite g5` | FR-GATE-07, FR-GATE-06, FR-GATE-11 | Gated sessions receive 0 known-format canaries, also inside base64 payloads. Each control shows whether that path leaks without the gate. |
| Transport probes | `python bench/g2_probes.py` | FR-GATE-09 | No pass/fail. Compare the observations with the last committed run when an agent version changes. |

**Exit codes.** `canary_harness.py` exits 0 when it passed, 1 when a known-format canary reached the provider through a protected path, and 2 when the run is inconclusive (an agent is missing or did not finish). `gate_e2e` exits 0 unless the real home changed, so read every scenario in its report.

## 4. Benchmarks

| Command | What it reports |
|---|---|
| `python -m bench.gate_latency` | Latency the gate adds over direct-to-mock, with a pass-through scan. Accepts `--tls` and `--no-pool`. |
| `python -m bench.gate_scan_latency --recordings DIR/mock` | Replays recorded sessions through the real scanner. First turns, with a cold cache, are reported separately from follow-up turns. |
| `python -m bench.gate_encoded_fp --recordings DIR/mock` | Counts requests the FR-GATE-11 rule would block. Every hit must be a real canary. |

In these commands, `DIR` is the output directory of an end-to-end run (see Output above). NFR-01's target is 50 ms p95 added per request.

## 5. Not covered by the automated suites

- Live sessions against a real provider. `bench/live_canary_check.py` needs a logged-in agent and is run by hand.
- The fresh-machine setup by someone else (FR-READ-03).
- The one-time Codex `/hooks` trust.
- Paths that only exist on macOS.

Results of the latest verification run are in [the design doc's Progress section](design.md#progress-oct-21-2026).
