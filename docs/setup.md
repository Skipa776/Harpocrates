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
   - The `hooks` block adds a content-aware check, `python -m Harpocrates.hooks`, that runs before every built-in Read, content-mode Grep, and Bash command. It covers files with ordinary names, such as a token in `config/settings.py`.
     - It scans the target with the same detector and threshold as `safe_read`. If anything would be redacted, it refuses the call and tells the agent to use `safe_read` or `safe_grep`.
     - It prints only the path, the count, and the secret types, never a value.
     - Claude Code treats a crashed hook as "allow", so the hook catches every error and blocks the call instead.
     - It never sets a short timeout, because a timed-out hook allows the call.

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

**What the hook costs.**
- **Edits:** Claude Code allows Edit only on a file it has read with its built-in Read. So Claude can't edit a file the hook refuses, and needs to change it some other way or with your help.
- **How often:** on 1,000 random files from 16 open-source repos, the hook refused 30 (3.0%). Almost all were `generic` false alarms, such as a hash or a test fixture.
- **Speed:** about 0.05–0.15 s per call.

**What the hook follows in a shell command:**
- `sh -c` bodies, including after `sudo`, `env`, or `timeout`, plus `$(...)` and backticks;
- `cd` and `git -C`, globs, `~` and `$VAR` paths, and `if=FILE` arguments;
- `git show rev:path`, which it fetches read-only to scan;
- `grep`, `rg`, and `git grep` searches. Their pattern is never evaluated: attached flags, inversion (`-v`), and basic-regex syntax all change which lines print. Every file in scope counts unless the search prints only names (`-l`, `--files`, `-c`). In a repo with a flagged file, a shell search is refused, and the agent uses the Grep tool or `safe_grep`;
- `env`, `printenv`, and `$VAR` references to environment variables that hold secrets. The refusal names the variable, never its value.

**What still gets through until the egress gate ships:** a script that opens files itself (`python x.py`, `node -e`), `find -exec`, `xargs` fed by a pipe, and `git diff` or `git log -p` output.

**The Grep tool.** Its input is structured, so the hook checks only the files its pattern matches. Install [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) so those matches are exactly what Claude's Grep would find. Without it, the hook uses Python's regex engine in a child process with a 5-second limit. A pattern it can't evaluate the same way counts as matching every file in scope, and so does a timeout: grep's basic-regex syntax and POSIX classes like `[[:alpha:]]` both fall into this. A search over more than 2,000 files is refused outright, and the agent should use `safe_grep` on a narrower path.

**Failure behavior.** Any error, an unparseable command, or passing its 25-second budget makes the hook block the call. It never fails open.

**Check:** in a repo with a `.env`, ask Claude to show the file. The built-in Read is refused, and the answer contains placeholders instead of values.

## 4. Codex CLI

1. **Merge the printed TOML into `~/.codex/config.toml`.** It does three things:
   - It registers `[mcp_servers.harpocrates]`.
   - It defines a `harpocrates` permission profile that extends `:workspace`. The profile denies the home-directory credential stores and the file patterns under every workspace root.
   - It makes that profile the default with `default_permissions = "harpocrates"`. That key must sit above the first `[table]` in the file. If you already use a profile, add the deny lines to yours instead.

   Codex reads files with shell commands, and the profile applies to every sandboxed command. Codex doesn't apply profiles to MCP servers, so `safe_read` still reads the denied paths.

2. **Add the printed note to `~/.codex/AGENTS.md`.**

3. **Trust the hook.** The printed config also adds a `PreToolUse` hook for `Bash`. It is the same content-aware check as Claude Code's: before a shell command, it refuses commands that would print a file holding a secret, including ordinarily named files like `config/settings.py`. Codex runs a new or changed hook only after you review it, so start `codex`, type `/hooks`, and trust the Harpocrates hook.

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
| Claude Code 2.1.286, live, "print every file" prompt (`bench/live_canary_check.py --agent claude`) | 0 of 15 canaries leaked. The agent read all 8 files through `safe_read`. |
| Claude Code, live, routine task through the built-in Read (`--task`) | 0 of 15 leaked. The hook refused all 4 Reads, and the agent retried with `safe_read`. |
| Claude Code, live, told to use only Read or `cat` (`--builtin`) | 0 of 15 leaked. `README.md` was read normally, and the hook refused Read on the 4 secret-bearing files. |
| Claude Code, live, same task without Harpocrates (`--task --off`, control) | 8 of 15 leaked, every canary in those 4 files. This shows the check can detect a leak. |
| Leak detection in `bench/live_canary_check.py` | Exact value, any 12-character run, base64, hex, and URL-encoded forms. A run where the agent did nothing reports INCONCLUSIVE, never "0 leaked". |
| Codex 0.159.2, live, routine task, path rules only (hook not yet trusted) | 8 of 15 leaked: the same files, read with `nl` in one `zsh -lc` loop |
| Codex hook, live | Pending your one-time trust in `/hooks`. Then run `python bench/live_canary_check.py --agent codex --task --installed`. Its logic is unit-tested on that exact command shape. |
| Fresh-machine setup in under 5 minutes, done by someone else | Pending. This is FR-READ-03's acceptance test. |
