"""The scan step (FR-GATE-03): the one place a request body is judged before it can leave.

G3 ships only a stub that raises, so every request is blocked. G4 replaces the stub with the real scanner.
"""
from __future__ import annotations

import threading
from typing import Callable, NamedTuple, Union

from Harpocrates.gate.errors import is_block_code
from Harpocrates.gate.routes import Route

SCAN_ERROR = "HARPO-BLOCK-SCAN-ERROR"
SCAN_TIMEOUT = "HARPO-BLOCK-SCAN-TIMEOUT"


class Forward(NamedTuple):
    """
    Description: A scan step's answer that the request may go on.
    Parameters:
        body: the exact bytes to forward (decoded, possibly rewritten).
    Output: An immutable record.
    """

    body: bytes


class Block(NamedTuple):
    """
    Description: A scan step's answer that the request must not go on.
    Parameters:
        reason: a code such as "HARPO-BLOCK-UNKNOWN-TYPE" (see errors.is_block_code).
    Output: An immutable record.
    """

    reason: str


class RequestHead(NamedTuple):
    """
    Description: What the scan step is told about a request besides its body (decision 12): the header values, the path
        and the query, which can hold a secret as well.
    Parameters:
        headers: (name, value) pairs as received, in order, duplicates and case kept, without the gate token header.
        path: the request path without the query, as received (for example "/v1/messages").
        query: the query string without the leading "?", or "" when there is none.
    Output: An immutable record built by the server.
    """

    headers: tuple[tuple[str, str], ...]
    path: str
    query: str


ScanStep = Callable[[bytes, Route, RequestHead], Union[Forward, Block]]


class ScanUnavailableError(RuntimeError):
    """
    Description: Raised by the default scan step, which exists only so that nothing is forwarded before G4.
    Parameters: None
    Output: An exception; run_scan turns it into a block.
    """


def default_scan_step(body: bytes, route: Route, head: RequestHead) -> Forward:
    """
    Description: The stub scan step G3 ships with. It never lets a request through.
    Parameters:
        body: the decoded request body (ignored).
        route: the route of the request (ignored).
        head: the headers, path and query of the request (ignored).
    Output: Never returns. Raises ScanUnavailableError on every call.
    """
    raise ScanUnavailableError("the scan step arrives in G4; until then nothing is forwarded")


def run_scan(step: ScanStep, body: bytes, route: Route, head: RequestHead, timeout: float) -> Union[Forward, Block]:
    """
    Description: Run a scan step on a decoded body, treating anything but a clear answer as a block.
    Parameters:
        step: the scan step, called as step(body, route, head).
        body: the decoded request body.
        route: the route of the request.
        head: the headers, path and query of the request.
        timeout: seconds the step may take.
    Output: The step's Forward when it returned Forward with bytes. Its Block when the reason passes
        errors.is_block_code. Otherwise Block("HARPO-BLOCK-SCAN-TIMEOUT") when the step ran past the timeout (it is
        not waited for, and a late answer is dropped), or Block("HARPO-BLOCK-SCAN-ERROR") for an exception, a Block with
        an unsafe reason, or an answer of any other kind.
    """
    answer: list[object] = []

    def work() -> None:
        """
        Description: Call the step and keep what it returned or raised.
        Parameters: None
        Output: None. Appends one item to answer.
        """
        try:
            answer.append(step(body, route, head))
        except BaseException as error:  # a step that fails in any way must still end in a block
            answer.append(error)

    worker = threading.Thread(target=work, name="harpo-gate-scan", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return Block(SCAN_TIMEOUT)
    result = answer[0] if answer else None
    if isinstance(result, Forward) and isinstance(result.body, bytes):
        return result
    if isinstance(result, Block) and isinstance(result.reason, str) and is_block_code(result.reason):
        return result
    return Block(SCAN_ERROR)
