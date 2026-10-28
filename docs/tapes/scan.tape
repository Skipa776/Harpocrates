# DOC-DEMO-01 / DOC-DEMO-02 scene 1: harpo scan finds secrets a regex scanner misses.
# Render from the repository root:  vhs docs/tapes/scan.tape
# Everything is a generated canary (bench/canary_repo.py). Nothing real is read, and HOME is a throwaway directory.
Output docs/assets/demo-scan.gif

Require gitleaks

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
Type `if ! command -v harpo >/dev/null; then harpo() { python -m Harpocrates.cli "$@"; }; fi` Enter
Type `D=$(mktemp -d /tmp/demo.XXXXXX); python $ROOT/bench/canary_repo.py $D/repo >/dev/null; cd $D/repo` Enter
Type `clear` Enter
Show

Type `# 15 generated canary secrets in 8 files. One file, two scanners:` Enter
Sleep 500ms
Type `gitleaks dir .env --no-banner --redact -v 2>&1 | grep -E 'RuleID|Line:|leaks found'` Enter
Sleep 3s
Type `harpo scan .env` Enter
Sleep 4s
Type `# whole repo: harpo found 14 of 15 canaries, gitleaks 11 of 15 (details in the README)` Enter
Sleep 3s
