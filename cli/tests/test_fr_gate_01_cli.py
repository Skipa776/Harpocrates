"""FR-GATE-01 / SEC-03 / FR-GATE-03: `python -m Harpocrates.gate --port N --token-file F` as a real process.

The gate here is the shipped one with its default scan step, so it can only ever refuse: no request reaches a provider.
The token file and every token are made up by the test.
"""
from __future__ import annotations

import http.client
import json
import signal
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from Harpocrates.gate.session_token import TOKEN_HEADER, new_token, write_token_file


def start(child_python, *args: str) -> subprocess.Popen:
    return subprocess.Popen(child_python("Harpocrates.gate", *args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def listening_url(proc: subprocess.Popen) -> str:
    """Read the one JSON line the gate prints once it serves; kill the process if it never does."""
    timer = threading.Timer(20, proc.kill)
    timer.start()
    try:
        line = proc.stdout.readline()
    finally:
        timer.cancel()
    return json.loads(line)["listening"]


def stop(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> tuple[int, str]:
    proc.send_signal(sig)
    out, err = proc.communicate(timeout=15)
    return proc.returncode, err


@pytest.fixture
def token_file(tmp_path: Path) -> tuple[Path, str]:
    path, token = tmp_path / "gate.token", new_token()
    write_token_file(path, token)
    return path, token


def request(url: str, method: str, target: str, body: bytes | None, headers: dict) -> tuple[int, bytes]:
    host, port = url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        conn.request(method, target, body=body, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_fr_gate_01_the_command_serves_on_loopback_with_the_stub_scan_step_and_stops_cleanly(child_python, token_file) -> None:
    path, token = token_file
    proc = start(child_python, "--port", "0", "--token-file", str(path))
    try:
        url = listening_url(proc)
        assert url.startswith("http://127.0.0.1:") and int(url.rsplit(":", 1)[1]) > 0
        # No token: 401. Right token: the stub scan step refuses, so nothing is forwarded.
        assert request(url, "POST", "/v1/messages", b"{}", {"anthropic-version": "x"})[0] == 401
        for method, target, body in (("POST", "/v1/messages", b"{}"), ("POST", "/v1/responses", b"{}"),
                                     ("POST", "/v1/messages/count_tokens", b"{}"), ("GET", "/v1/models", None)):
            status, payload = request(url, method, target, body, {TOKEN_HEADER: token})
            assert status == 400 and b"HARPO-BLOCK-SCAN-ERROR" in payload
    finally:
        code, err = stop(proc)
    assert code == 0
    lines = [json.loads(line) for line in err.splitlines() if line.strip()]
    assert [e["decision"] for e in lines] == ["blocked"] * 5 and lines[0]["reason"] == "HARPO-BLOCK-TOKEN"
    assert token not in err  # the token is never written to the log


def test_fr_gate_01_sigint_also_stops_it_cleanly(child_python, token_file) -> None:
    path, _ = token_file
    proc = start(child_python, "--token-file", str(path))
    listening_url(proc)
    assert stop(proc, signal.SIGINT)[0] == 0


def test_fr_gate_01_the_command_has_no_way_to_bind_another_address(child_python, token_file) -> None:
    path, _ = token_file
    proc = start(child_python, "--host", "0.0.0.0", "--token-file", str(path))
    out, err = proc.communicate(timeout=20)
    assert proc.returncode == 2 and out == "" and "--host" in err


@pytest.mark.parametrize("make", ["missing", "loose", "short", "empty"])
def test_sec_03_an_unusable_token_file_stops_the_gate_before_it_listens(child_python, tmp_path: Path, make: str) -> None:
    path = tmp_path / "gate.token"
    if make != "missing":
        write_token_file(path, new_token())
        if make == "loose":
            path.chmod(0o644)
        elif make == "short":
            path.write_text("tooshort\n")
        else:
            path.write_text("")
    proc = start(child_python, "--token-file", str(path))
    out, err = proc.communicate(timeout=20)
    assert proc.returncode == 2 and out == "" and "token" in err.lower()
    assert "Traceback" not in err


def test_fr_gate_01_a_bad_port_or_no_token_file_option_is_a_usage_error(child_python, token_file) -> None:
    path, _ = token_file
    for args in (["--port", "70000", "--token-file", str(path)], ["--port", "abc", "--token-file", str(path)], ["--port", "0"]):
        proc = start(child_python, *args)
        out, err = proc.communicate(timeout=20)
        assert proc.returncode == 2 and out == ""


def test_fr_gate_01_a_port_already_in_use_exits_1(child_python, token_file) -> None:
    path, _ = token_file
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        proc = start(child_python, "--port", str(taken.getsockname()[1]), "--token-file", str(path))
        out, err = proc.communicate(timeout=20)
    assert proc.returncode == 1 and out == "" and "127.0.0.1" in err and "Traceback" not in err


def test_fr_gate_01_main_reports_its_exit_status_in_process(token_file, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from Harpocrates.gate.__main__ import main

    path, token = token_file
    assert main(["--token-file", str(tmp_path / "missing.token")]) == 2
    assert main(["--port", "70000", "--token-file", str(path)]) == 2
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        assert main(["--port", str(taken.getsockname()[1]), "--token-file", str(path)]) == 1
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err and "Traceback" not in captured.err
