# Harpocrates

Harpocrates Guard is a local security boundary that stops secrets from leaving a developer's machine through AI coding agents (Claude Code, Codex). A shared detection core (regex, entropy, ONNX) backs three layers: an MCP read tool that redacts file reads, an egress proxy that scans every provider-bound request and fails closed, and the existing commit-time scanner. Spec: [docs/design.md](docs/design.md).

## Layout

`core/` Rust detection engine · `cli/` Python package (`Harpocrates`) and tests · `bench/` benchmarks · `scripts/` training and data tooling · `docs/` spec, data provenance, and assets. `read/`, `gate/`, `adapters/`, and `inspect/` are created in the milestone that needs them.

## Commands

```bash
pip install -e ".[dev,train,mcp]"
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

## Current milestone

**M1: Read tool** (Oct 6 – Oct 19). Done: FR-READ-01, FR-READ-02, SEC-02, FR-CORE-02, FR-CORE-03, SEC-06, FR-INSTALL-03. FR-READ-03 shipped with a content-aware read hook. Exit criterion met live on Claude Code (0/15 canaries); the Codex hook awaits one-time /hooks trust, and the fresh-machine setup by someone else is pending. Next: M2, egress gate.
