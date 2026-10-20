"""FR-GATE-01, SEC-03, FR-INSTALL-04, FR-INSTALL-05, FR-GATE-03, FR-GATE-06: the gate end to end, in front of
bench/mock_provider.py on 127.0.0.1. These are the thirteen tests of the G3 spec "Test plan", plus a few that pin
what the plan implies. Every credential and canary in them is a made-up string."""
from __future__ import annotations

import dataclasses
import socket

import pytest
from gate_rig import passthrough_scan, running_rig  # noqa: F401

from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import new_token


def test_fr_gate_01_binds_loopback_only() -> None:
    with running_rig() as rig:
        assert rig.gate.host == "127.0.0.1" and rig.gate.server.server_address[0] == "127.0.0.1"
        assert rig.gate.port > 0 and rig.gate.url == f"http://127.0.0.1:{rig.gate.port}"
        # Not reachable on this machine's other addresses.
        try:
            other = socket.gethostbyname(socket.gethostname())
        except OSError:
            other = "127.0.0.1"
        if other != "127.0.0.1" and not other.startswith("127."):
            with pytest.raises(OSError):
                socket.create_connection((other, rig.gate.port), timeout=2).close()
    # The settings refuse any other address, and start_gate re-checks the settings it is given.
    for host in ("0.0.0.0", "::", "::1", "localhost", "192.0.2.1", ""):
        with pytest.raises(ValueError, match="127.0.0.1"):
            GateConfig(host=host)
    forged = GateConfig()
    object.__setattr__(forged, "host", "0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        start_gate(forged, new_token())
    assert dataclasses.is_dataclass(forged)
