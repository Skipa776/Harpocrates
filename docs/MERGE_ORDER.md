# PRD-01 CLI Restoration — Merge Order

The README updates on `task/8-readme-cli-docs` document the post-merge state of
the CLI restoration work. The flags and modules referenced (`--show-secrets`,
`--fail-on`, `Harpocrates.utils.redaction`, `--ml-threshold` default `0.19`)
land in separate PRs and must merge **before** this docs PR.

## Required merge order

1. `task/1-show-secrets-flag` — adds `--show-secrets` to `scan`
2. `task/2-fail-on` — adds `--fail-on <severity>` and the severity gate
3. `task/4-redaction-helper` — extracts `Harpocrates.utils.redaction`
4. `task/9-ml-threshold-default` — sets the `--ml-threshold` default to `0.19`
5. `task/8-readme-cli-docs` — **this PR** (documents 1–4)

## Why a separate docs PR?

Each implementation PR keeps a tight diff focused on one behavioral change
plus its tests. The cross-cutting CLI Reference / Quick-Start / Exit Codes
update touches a single file (the README) and benefits from being reviewed
as one coherent narrative rather than scattered across four PRs.

## Rebase plan if merged out of order

If a reviewer prefers to merge `task/8` first, rebase it on top of any
integration branch that contains tasks 1, 2, 4, and 9. The README diff
itself does not depend on any specific commit ordering — it only depends
on the union of the four feature branches.
