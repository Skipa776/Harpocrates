# harpocrates-prompt

A Claude Code plugin that draws the Harpocrates "edit without secret
protection" question as a styled card.

Harpocrates keeps files that contain secrets out of the model's context: Claude
reads them through `mcp__harpocrates__safe_read`, which replaces each secret with
a placeholder. When Claude wants to **edit** such a file with its built-in tools,
the Harpocrates hook asks first (reads of secret files never ask). This plugin
replaces the plain permission text with a card that says which file, which kinds
of secret, and what allowing means, above a two-option choice.

The plugin only changes how the question looks and how the answer is read. It
never weakens the hook's verdict:

- Only the exact choice **Allow edits this session** allows the edit.
- **Keep secrets protected**, a dismissed question, a typed answer under "Other"
  and any error all deny, and the denial tells Claude to use `safe_read` and to
  ask you to make the edit.
- The plugin draws its card only for a real Edit, Write, MultiEdit or
  NotebookEdit call whose own file path is the one the hook's reason names, and
  only for a reason that matches the hook's wording exactly. For anything else (a
  different tool, a different path, a changed message, a verdict from a settings
  rule, a query rather than a real call) it steps aside and Claude Code shows its
  own dialog.
- The card shows the tool and the target path, and nothing else from the tool's
  input: the text being written is never drawn. Control, bidirectional and
  zero-width characters in a path are shown as `?`.

Other plugins that hook `AskUserQuestion` could answer this question on your
behalf, so install only plugins you trust.

## Requirements

The Harpocrates CLI hook must be installed, because it is what produces the
verdict this plugin asks about. The PyPI release that carries this hook is
pending; until it is out, install the CLI from source:

```
pip install git+https://github.com/Skipa776/harpocrates
harpocrates install
```

Without it the plugin does nothing.

## Install

In a Claude Code terminal session:

```
/plugin marketplace add Skipa776/harpocrates
/plugin install harpocrates-prompt --marketplace Skipa776/harpocrates
```

Then choose a scope. The plugin is active in that session straight away, and
under the user scope in every session after.

## What it looks like

```
╭──────────────────────────────────────────────────╮
│ 🔒 Harpocrates — edit without secret protection  │
│ Edit  config/settings.py                         │
│ Claude wants to edit config/settings.py, which   │
│ contains secrets (api_token, password).          │
│ Allowing lets Claude read and edit this file     │
│ with its built-in tools for the rest of this     │
│ session; these values go to the model provider   │
│ as-is.                                           │
│ Claude can still read it with safe_read without  │
│ this.                                            │
╰──────────────────────────────────────────────────╯
Allow Claude to edit settings.py without secret protection?

 1. Keep secrets protected
    Claude keeps using safe_read; the file stays unedited
 2. Allow edits this session
    Built-in Read and Edit of this file, until the session ends
```

A call made by a subagent adds ` · from a subagent` to the header. The card is
kept inside the engine's row limit: when space is short the last line (the dim
hint) goes first, then part of the call line, then the left end of a long path.

## Develop

```
claude plugin validate plugins/harpocrates-prompt
claude plugin test plugins/harpocrates-prompt
```

The tests cover parsing the reason (paths with spaces and parentheses, the
`unscanned` kind, CRLF, garbage), each answer, the cases where the plugin steps
aside, the subagent label, fitting the card at narrow and wide viewports, and
that no tool input except the target path is ever drawn.

## License

MIT, as the rest of the repository.
