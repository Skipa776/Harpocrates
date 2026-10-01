# Setup: route an agent's file reads through the read tool

FR-READ-03. This guide makes Claude Code or Codex refuse their built-in reads of files that usually hold secrets. The agent reads those files through the Harpocrates MCP tools instead: `safe_read` and `safe_grep` return them with every detected secret replaced by a `<<HARPO:type:hash>>` placeholder.

What this protects today, and what it doesn't, is in the [threat model](threat-model.md). In short, until the egress gate ships (M2), a secret the agent reaches through the shell in some other way still goes to the model.

## 1. Install

```bash
pip install "harpocrates[mcp]"
harpocrates setup            # lists detected harnesses and their support tier
```

`harpocrates setup claude-code` or `harpocrates setup codex` prints the exact config for this machine. That config includes the path of the Python interpreter that runs the MCP server, so it keeps working from a virtual environment and from agents launched from an IDE. The sections below explain what that config does. Writing it into the harness's files for you, with a backup, comes with the M2 installer (FR-INSTALL-01).

## 2. Paths that are denied

The single list lives in `cli/Harpocrates/harness.py`, and tests check that both harnesses' config covers all of it.

- **Any file, at any depth:** `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ecdsa*`, `id_ed25519*`, `*.tfvars`, `*.tfstate`, `*.jks`, `*.keystore`, `credentials.json`, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`
- **Whole directories:** `~/.ssh`, `~/.aws`, `~/.azure`, `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.gnupg`

**The trade-off is deliberate.** The list also catches files that are usually safe, such as `.env.example`, `id_rsa.pub`, a registry-only `.npmrc`, or an unrelated `*.key` file. Blocking them costs little, because the agent still reads them through `safe_read` and gets the same text back with nothing to redact. A narrower list would let a real secret file through the built-in tools.

## 3. Claude Code

1. **Register the read tool for every project:**

   ```bash
   claude mcp add --scope user harpocrates -- "$(python -c 'import sys; print(sys.executable)')" -m Harpocrates.mcp.server
   ```

2. **Merge the printed `permissions` block into `~/.claude/settings.json`.**
   - It denies `Read(//**/.env)` and the rest of the list. The `//` prefix anchors at the filesystem root, so the rules cover every project.
   - It allows `mcp__harpocrates__safe_read` and `mcp__harpocrates__safe_grep`, so the agent can use them without prompting.
   - The deny rules also apply to Grep, Glob, `@file` mentions, and the `cat`, `head`, `tail`, and `sed` commands that Claude Code recognizes in Bash.

3. **Add the printed note to `~/.claude/CLAUDE.md`.** When a read is refused, the agent then knows to use `safe_read`.

4. **Optional hardening.** Claude Code's deny rules don't cover `grep -r` run inside the directory, or a script that opens files itself. The Bash sandbox does: it enforces the block at the OS level for every Bash command and its child processes. The sandbox doesn't apply to MCP servers, so `safe_read` keeps working.

   ```json
   {
     "sandbox": {
       "enabled": true,
       "filesystem": {
         "denyRead": ["~/.ssh", "~/.aws", "~/.azure", "~/.config/gcloud", "~/.kube", "~/.docker",
                      "~/**/.env", "~/**/.env.*", "~/**/*.pem", "~/**/*.key", "~/**/id_rsa*"]
       }
     }
   }
   ```

   The example lists only part of the list in section 2; extend it with the rest. Turning on the sandbox changes how every shell command runs. Try it in one project first with `/sandbox`.

**Check:** in a repo with a `.env`, ask Claude to show the file. The built-in Read is refused, and the answer contains placeholders instead of values.

## 4. Codex CLI

1. **Merge the printed TOML into `~/.codex/config.toml`.** It does three things:
   - It registers `[mcp_servers.harpocrates]`.
   - It defines a `harpocrates` permission profile that extends `:workspace`. The profile denies the home-directory credential stores and the file patterns under every workspace root.
   - It makes that profile the default with `default_permissions = "harpocrates"`. That key must sit above the first `[table]` in the file. If you already use a profile, add the deny lines to yours instead.

   Codex reads files with shell commands, and the profile applies to every sandboxed command. Codex doesn't apply profiles to MCP servers, so `safe_read` still reads the denied paths.

2. **Add the printed note to `~/.codex/AGENTS.md`.**

**Check:**

```bash
codex mcp list                      # harpocrates ... enabled
cd a-repo-with-a-dotenv
codex sandbox -- cat .env           # cat: .env: Operation not permitted
```

**Limit:** the profile is off when Codex runs unsandboxed, with `sandbox_mode = "danger-full-access"` or `--dangerously-bypass-approvals-and-sandbox`.

## What was verified

| Check | Result |
| --- | --- |
| Codex 0.159.2 loads the generated config | Yes. An invalid profile is rejected, so the check is real. |
| Codex sandbox, generated profile: `cat .env`, `cat sub/app.pem`, `ls ~/.ssh`, `ls ~/.aws` | All blocked; `cat readme.txt` still works |
| Codex sandbox, default config (control) | `.env` and `~/.ssh` readable |
| Claude Code 2.1.286 rule syntax | Matches the [permissions docs](https://code.claude.com/docs/en/permissions); not yet exercised in a live session |
| Fresh-machine setup in under 5 minutes, done by someone else | Pending. This is FR-READ-03's acceptance test. |
