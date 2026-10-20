"""A made-up machine for the SEC-10 tests: a home folder with an install record that says the gate is on, user
settings that point Claude Code at it, an empty repository and an empty managed-settings folder.

Nothing here is a real credential; the token value is a placeholder that no test may ever see in a message.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from Harpocrates import gatecheck

PORT = 47321
GATE_URL = f"http://127.0.0.1:{PORT}"
HEADER = "x-harpo-gate-token"
TOKEN_VALUE = "not-a-real-token-0123456789"
HEADERS_OK = f"{HEADER}: {TOKEN_VALUE}"
EVIL = "https://proxy.attacker.example:8443"
EVIL_SECRET_URL = "https://user:hunter2@proxy.attacker.example:8443/path/with-secret-4242?token=abc"
WATCHED = ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_CUSTOM_HEADERS",
           "CLAUDE_PROJECT_DIR")


def write_json(path: Path, data: object) -> Path:
    """Write data as indented JSON, creating folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def line_of(path: Path, key: str) -> int:
    """The 1-based line of the first line that holds the quoted key."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if f'"{key}"' in line:
            return number
    raise AssertionError(f"{key} not in {path}")


@dataclass
class World:
    """The made-up machine."""

    home: Path
    repo: Path
    managed: Path

    @property
    def record(self) -> Path:
        return self.home / ".harpo" / "install.json"

    @property
    def user_settings(self) -> Path:
        return self.home / ".claude" / "settings.json"

    @property
    def project_settings(self) -> Path:
        return self.repo / ".claude" / "settings.json"

    @property
    def local_settings(self) -> Path:
        return self.repo / ".claude" / "settings.local.json"

    def set_record(self, gate: object = ..., **top: object) -> Path:
        """Write the install record; `gate` is the gate entry (leave it out for the default, None for no entry)."""
        data: dict = {"version": 1, **top}
        entry = {"enabled": True, "port": PORT, "token_header": HEADER} if gate is ... else gate
        if entry is not None:
            data["gate"] = entry
        return write_json(self.record, data)

    def set_user(self, env: Optional[dict] = None) -> Path:
        """Write ~/.claude/settings.json; the default env is what a working gate install leaves there."""
        env = {"ANTHROPIC_BASE_URL": GATE_URL, "ANTHROPIC_CUSTOM_HEADERS": HEADERS_OK} if env is None else env
        return write_json(self.user_settings, {"env": env})

    def set_project(self, env: dict, local: bool = False) -> Path:
        return write_json(self.local_settings if local else self.project_settings, {"env": env})

    def set_managed(self, env: dict, name: str = "managed-settings.json") -> Path:
        return write_json(self.managed / name, {"env": env})


def make_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """A machine where the gate is on and everything points at it; the test then breaks one thing. Each test file
    wraps this in its own `world` fixture."""
    home = Path(os.environ["HOME"])  # the autouse conftest fixture made it
    repo = tmp_path / "repo"
    repo.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setattr(gatecheck, "managed_dir", lambda: managed)  # never the real /Library/Application Support/...
    for name in WATCHED:
        monkeypatch.delenv(name, raising=False)
    machine = World(home=home, repo=repo, managed=managed)
    machine.set_record()
    machine.set_user()
    return machine
