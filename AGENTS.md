# Harpocrates

Harpocrates Guard is a local security boundary that stops secrets from leaving a developer's machine through AI coding agents (Claude Code, Codex). A shared detection core (regex, entropy, ONNX) backs three layers: an MCP read tool that redacts file reads, an egress proxy that scans every provider-bound request and fails closed, and the existing commit-time scanner. Spec: [docs/design.md](docs/design.md).

## Layout

`core/` Rust detection engine · `cli/` Python package (`Harpocrates`) and tests · `bench/` benchmarks · `scripts/` training and data tooling · `docs/` spec, data provenance, and assets. `read/`, `gate/`, `adapters/`, and `inspect/` are created in the milestone that needs them.

## Commands

```bash
pip install -e ".[dev,explain,mcp]"
cargo build --release --manifest-path core/Cargo.toml
python -m pytest -q
ruff check .
cargo fmt --manifest-path core/Cargo.toml --check && cargo clippy --manifest-path core/Cargo.toml --all-targets -- -D warnings
```

## Working rules

1. Work on one requirement ID at a time. Write its acceptance test first, named after the ID (for example `test_fr_gate_03_fail_closed`).
2. Never weaken a security requirement to make a test pass. Never switch fail-closed to fail-open, and never log secret values. Stop and ask instead.
3. Use only generated canary secrets in tests and fixtures. Never put a real credential in the repo or a prompt.
4. Any change to thresholds, placeholder format, or logging needs a note explaining the trade-off.
5. Never use benchmark suites or evasion cases for training or tuning.
6. Every pull request lists the requirement IDs it closes and the tests that prove them.
7. Before a feature counts as done, an end-to-end test exercises it through its real entry point: the `harpo` CLI, the hook as the agent invokes it, or the gate with a real `claude -p` or `codex exec` session in a throwaway home against the mock provider. Unit and component tests alone do not finish a feature.

## Current milestone

**R0: Cleanup, docs and rename** (done Oct 19, branch `r0/cleanup`, see `docs/cleanup-report.md`). M1 is done except the fresh-machine setup by someone else and the Codex `/hooks` trust. Next: M2, the egress gate, then the M3 proof requirements. M4 to M8 are in `docs/design.md` under "Roadmap after R0", awaiting Josh's review; M8 is deferred. Five parts are Josh's to write by hand (marked **(Josh writes)** there): write only specs, interfaces and fixtures for them.

**M1: Read tool** (Oct 6 – Oct 19). Done: FR-READ-01, FR-READ-02, SEC-02, FR-CORE-02, FR-CORE-03, SEC-06, FR-INSTALL-03. FR-READ-03 shipped with a content-aware read hook. Exit criterion met live on Claude Code (0/15 canaries); the Codex hook awaits one-time /hooks trust, and the fresh-machine setup by someone else is pending. Next: M2, egress gate.
