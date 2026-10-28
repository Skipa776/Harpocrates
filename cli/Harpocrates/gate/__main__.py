"""Run the gate: python -m Harpocrates.gate --port N --token-file F [--audit-dir D]

There is no bind-address option (the gate listens on 127.0.0.1 only) and no way to skip the scan: no override flag exists
(FR-GATE-03). `harpo guard up` runs this command with --audit-dir ~/.harpo (G6, G7).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from Harpocrates.gate.config import GateConfig
from Harpocrates.gate.server import start_gate
from Harpocrates.gate.session_token import read_token_file


def _log_to_stderr(line: str) -> None:
    """
    Description: Write one request-log line to stderr and flush it.
    Parameters:
        line: a JSONL line from the request log.
    Output: None.
    """
    sys.stderr.write(line)
    sys.stderr.flush()


def main(argv: Optional[list[str]] = None) -> int:
    """
    Description: Start the gate and serve until SIGINT or SIGTERM.
    Parameters:
        argv: command-line arguments, or None to use sys.argv.
    Output: Exit status 0 after a clean stop; 2 for a usage error, a bad port or a token file that is missing, loose or
        weak (a message on stderr, nothing listening, never the token); 1 when the port cannot be bound. Prints one JSON
        line {"listening": URL} to stdout once it serves, and one request-log line per request to stderr.
    """
    parser = argparse.ArgumentParser(prog="python -m Harpocrates.gate", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=0, help="TCP port on 127.0.0.1; 0 lets the system choose")
    parser.add_argument("--token-file", type=Path, required=True,
                        help="file holding the session token, mode 0600 (see session_token.write_token_file)")
    parser.add_argument("--audit-dir", type=Path, default=None,
                        help="folder for findings.jsonl and proof.jsonl; without it the gate writes neither")
    args = parser.parse_args(argv)
    try:
        token = read_token_file(args.token_file)
        config = GateConfig(port=args.port)
    except ValueError as error:
        print(f"harpo-gate: {error}", file=sys.stderr)
        return 2
    try:
        gate = start_gate(config, token, log_sink=_log_to_stderr, audit_dir=args.audit_dir)
    except OSError as error:
        print(f"harpo-gate: cannot bind {config.host}:{config.port}: {error.strerror or type(error).__name__}",
              file=sys.stderr)
        return 1
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    print(json.dumps({"listening": gate.url}), flush=True)
    stop.wait()
    gate.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
