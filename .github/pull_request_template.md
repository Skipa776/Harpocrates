## Summary

<One paragraph: what changed and why.>

## Requirements

<The requirement IDs this PR closes or advances, for example FR-GATE-03.>

## Tests

One line per new or changed test. Josh ticks each one after reading it.

- [ ] `test_fr_xxx_yy_name` (FR-XXX-YY): what it proves

## End-to-end evidence

<The real entry point exercised, the command, and its result: counts, not just an exit code.>

## Checks

- `ruff check .`
- `python scripts/check_docstrings.py --enforce`
- `pytest -q -m "not integration" --cov` (coverage floor 90%)
