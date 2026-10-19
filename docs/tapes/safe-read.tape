# DOC-DEMO-01 / DOC-DEMO-02 scene 2: an agent's read of .env is blocked, and safe_read returns placeholders.
# Render from the repository root:  vhs docs/tapes/safe-read.tape
# The agent calls are scripted (a hook event piped in, safe_read called directly), not a live session.
# Everything is a generated canary (bench/canary_repo.py), and HOME is a throwaway directory.
Output docs/assets/demo-safe-read.gif

Set Shell "bash"
Set FontSize 16
Set Width 1240
Set Height 640
Set Padding 20
Set Framerate 15
Set TypingSpeed 25ms
Set Theme "Dracula"

Hide
Type `export PS1='$ ' ROOT=$PWD HOME=$(mktemp -d /tmp/home.XXXXXX)` Enter
Type `PY=$([ -x $ROOT/.venv/bin/python ] && echo $ROOT/.venv/bin/python || command -v python); python() { "$PY" "$@"; }` Enter
Type `D=$(mktemp -d /tmp/demo.XXXXXX); python $ROOT/bench/canary_repo.py $D/repo >/dev/null; cd $D/repo` Enter
Type `printf '{"hook_event_name":"PreToolUse","tool_name":"Read","cwd":"%s","tool_input":{"file_path":"%s/.env"}}' $PWD $PWD > $D/event.json` Enter
Type `clear` Enter
Show

Type `# Scripted agent call, not a live session. The agent tries to Read .env:` Enter
Sleep 500ms
Type `python -m Harpocrates.hooks < $D/event.json` Enter
Sleep 4s
Type `clear` Enter
Type `# The hook points it to safe_read, which returns the same file with placeholders:` Enter
Sleep 500ms
Type `python -c "from Harpocrates.mcp.server import safe_read; print(safe_read('.env'))"` Enter
Sleep 4s
