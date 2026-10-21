"""FR-NAME-02, security: the agent-cannot-change-the-allowlist guard (allowlist.touches) knows `harpo` exactly as
well as `harpocrates`. Renaming the command must not hand the agent a name the guard has never heard of.

The shapes below are the ones test_hooks_allowlist.py already covers for `harpocrates`, crossed with every command
name: quotes, backslashes, `$'...'`, braces, paths to the program, wrappers (sudo, env, xargs, subshells ...), and
the sub-command spelled in the same ways. Pure text matching, no file or HOME access.
"""
from __future__ import annotations

import fnmatch

import pytest

from Harpocrates import allowlist

# harpo-mcp has no `allow`, but a name the guard refuses for one program it refuses for all four.
NAMES = ("harpo", "harpocrates", "harpo-mcp", "harpocrates-mcp")


def _spellings(n: str) -> list[str]:
    """The program word as an agent might write it."""
    return [
        n,
        f"/usr/local/bin/{n}", f"./{n}", f"../.venv/bin/{n}", f"$HOME/.local/bin/{n}", f"~/.venv/bin/{n}",
        f"'{n}'", f'"{n}"',
        f"{n[0]}\\{n[1:]}", f"{n[:3]}''{n[3:]}", f'{n[:4]}""{n[4:]}',
        f"$'\\x{ord(n[0]):02x}{n[1:]}'", f"$'\\{ord(n[0]):03o}{n[1:]}'", f"$'{n[:2]}\\u{ord(n[2]):04x}{n[3:]}'",
        f"{{{n},x}}", n.upper(),
    ]


_WRAPPERS = [
    "{c}", "sudo {c}", "sudo -u root {c}", "env A=1 {c}", "command {c}", "exec {c}", "nohup {c}", "time {c}",
    "timeout 5 {c}", "nice {c}", "! {c}", "echo x | xargs {c}", 'bash -c "{c}"', "sh -lc 'ls && {c}'", "ls && ({c})",
    "{ {c}; }", "if true; then {c}; fi", "for i in 1; do {c}; done", "cd / && {c}", "echo $({c})", "echo `{c}`",
    "ls; {c} &", "{c} &", "cd /tmp\n{c}",
]
_TAILS = [
    "allow x", "'allow' x", '"allow" x', "al\\low x", "a''llow x", "$'a\\x6clow' x", "--json allow x", "allow --list",
    "allow --name src --yes", "allow --remove /x", "ALLOW x", "$'\\141llow' x",
]


def _cmds(n: str) -> list[str]:
    plain = f"{n} allow x"
    return ([f"{s} allow x" for s in _spellings(n)] + [w.replace("{c}", plain) for w in _WRAPPERS]
            + [f"{n} {t}" for t in _TAILS])


@pytest.mark.parametrize("name", NAMES)
def test_fr_name_02_guard_refuses_allow_in_every_spelling_for_every_name(name: str) -> None:
    missed = [c for c in _cmds(name) if not allowlist.touches(c)]
    assert not missed, missed


def test_fr_name_02_guard_treats_harpo_exactly_like_harpocrates() -> None:
    """The same commands with the two names swapped get the same answer, whatever that answer is."""
    old, new = _cmds("harpocrates"), _cmds("harpo")
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert allowlist.touches(a) == allowlist.touches(b), (a, b)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("cmd", [
    "{n} \\\nallow x",  # a continuation between the words
    "{n}\\\n allow x",
    "{n} \\\n   allow x",
    "{a}\\\n{b} allow x",  # inside the program name: the shell joins the halves
    "{n} al\\\nlow x",  # inside `allow`
    "sudo {n} \\\n  allow /",
    "{n} --json \\\n  allow --name src",
])
def test_fr_name_02_guard_sees_through_a_line_continuation(name: str, cmd: str) -> None:
    """The shell removes backslash-newline before it reads words, so it can split the program from `allow`, or either
    word in two. (Found in review, for `harpocrates` too: the backslash was dropped but the newline stayed, which ended
    the match.)"""
    shown = cmd.format(n=name, a=name[:3], b=name[3:])  # a + b is the whole name
    assert allowlist.touches(shown), shown


@pytest.mark.parametrize("cmd", [
    "{n}\\\nallow x",  # no space: the shell reads ONE word, `{n}allow`
    "ls\\\n{n}",
    "{n} scan \\\n  --help",
])
def test_fr_name_02_guard_reads_a_continuation_without_space_as_one_word(cmd: str) -> None:
    assert not allowlist.touches(cmd.format(n="harpo")), cmd


def _globs(n: str) -> list[str]:
    """Patterns the shell would expand to `n` (and, for some, to its sibling programs: a glob that picks several
    files is refused too, since the guard cannot see which ones exist)."""
    return [n[:3] + "*", "*" + n[-3:], n[0] + "*" + n[-1], n[:-1] + "?", n[:-1] + "[" + n[-1] + "]",
            "[" + n[0] + "]" + n[1:], "?" + n[1:], n[:2] + "?" + n[3:]]


@pytest.mark.parametrize("name", NAMES)
def test_fr_name_02_guard_sees_through_a_glob_in_the_program_word(name: str) -> None:
    """`/usr/bin/harpocra*s allow /` runs the program: the shell expands the pattern to the file. Pattern and name
    are checked here with fnmatch, so a pattern that could not reach the program would fail the setup, not the guard."""
    patterns = _globs(name)
    assert all(fnmatch.fnmatchcase(name, p) for p in patterns), patterns
    for pattern in patterns:
        for prefix in ("", "./", "/usr/local/bin/", "$HOME/.venv/bin/", "'/opt/x'/"):
            for cmd in (f"{prefix}{pattern} allow /", f"sudo {prefix}{pattern} allow /",
                        f"bash -c '{prefix}{pattern} allow /'", f"{prefix}{pattern} --json allow /"):
                assert allowlist.touches(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "ls *.py",
    "ls harp*",  # a glob that names the program, but nothing runs it
    "ls harp* | grep allow",  # `allow` is another command's
    "grep -rn allow harp*",  # `allow` comes first
    "cat harp*.txt allow-notes.md",  # the pattern cannot match a program name
    "echo *.md allow",
    "ls foo* allow.txt",
    "rm -rf build/*; echo allow",
    "cat * | grep allow",
    "harp* scan .",  # no `allow`
    "ls [a-z]*.py",
    "ls ?? allow",  # a word of only wildcards is no mention of a program
    "cp -r src/* backup/ && echo allow",
    "grep -c allow harp*.txt",
    "tar czf x.tgz harp* && echo allow",
    'git commit -m "feat: x\\n\\n* allow list now knows harp names"',  # a bullet is a bare `*`
])
def test_fr_name_02_guard_leaves_globs_that_do_not_run_allow_alone(cmd: str) -> None:
    assert not allowlist.touches(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "harpoon allow x",
    "harpoons allow",
    "unharpo allow x",
    "harpo_x allow x",
    "harpo2 allow x",
    "xharpo allow x",
    "ls harpo",
    "cat harpo.txt",
    "ls ./harpo/",
    "grep harpo README.md",
    "echo harpo",
    "mv harpo.old harpo.new",
    "harpo scan .",
    "harpo scan --help",
    "harpo --version",
    "harpo install --dry-run",
    "harpo-mcp --help",
    "echo harpo | wc -c",
    "cd /Users/me/harpo && git log --oneline",
    "cd /Users/me/harpo && pytest -q",
    "grep -rn allow /Users/me/harpo/cli",  # `harpo/` is a folder, not the program
    "ls /srv/harpoon/allow",
])
def test_fr_name_02_guard_does_not_refuse_what_only_looks_like_harpo(cmd: str) -> None:
    """No over-block beyond what the guard already does for `harpocrates`."""
    assert not allowlist.touches(cmd), cmd
