"""Answers the gate makes itself (FR-GATE-01, decision 10): a provider-shaped error with a reason code and plain text."""
from __future__ import annotations

import json
import re

REASONS: dict[str, int] = {
    "HARPO-BLOCK-TOKEN": 401,
    "HARPO-BLOCK-UPGRADE": 426,
    "HARPO-BLOCK-ROUTE": 400,
    "HARPO-BLOCK-REQUEST": 400,
    "HARPO-BLOCK-BODY-FRAMING": 400,
    "HARPO-BLOCK-BODY-TOO-LARGE": 400,
    "HARPO-BLOCK-BODY-TRUNCATED": 400,
    "HARPO-BLOCK-ENCODING": 400,
    "HARPO-BLOCK-ENCODING-CORRUPT": 400,
    "HARPO-BLOCK-SCAN-ERROR": 400,
    "HARPO-BLOCK-SCAN-TIMEOUT": 400,
    "HARPO-BLOCK-INTERNAL": 400,
    "HARPO-UPSTREAM-FAILED": 502,
}  # every reason the gate itself gives; the scan step may add codes that pass is_block_code
UPSTREAM_FAILED = "HARPO-UPSTREAM-FAILED"
_BLOCK_CODE = re.compile(r"HARPO-BLOCK-[A-Z0-9]+(-[A-Z0-9]+)*")
_MAX_CODE_LENGTH = 48
# Phrases Claude Code matches in an error to start its own recovery (C1). Checked on the code with hyphens as spaces.
_RECOVERY_PHRASES = ("thinking", "signature", "bound to a different conversation", "extra inputs are not permitted",
                     "prompt is too long")
# One extra sentence for the codes where the user has something to do (FR-GATE-10: a secret the gate cannot rewrite).
_ADVICE = {"HARPO-BLOCK-PROVIDER-TEXT": "The conversation holds a secret in text that only the provider can rewrite. "
                                        "Start a new session. ",
           "HARPO-BLOCK-MEDIA-SECRET": "A file sent as base64 holds a secret. Send the file as text instead, so Harpo "
                                       "can redact it. ",
           "HARPO-BLOCK-ENCODED-SECRET": "Text in this request holds base64-encoded data with a secret in it. Remove it or "
                                         "send the decoded text, so Harpo can redact it. ",
           "HARPO-BLOCK-ENCODED-TOO-LARGE": "Text in this request holds more base64-encoded data than Harpo checks in one "
                                            "request. Remove the large encoded blob. "}
_KINDS = {
    "anthropic": {400: "invalid_request_error", 401: "authentication_error", 426: "invalid_request_error",
                  502: "api_error"},
    "openai": {400: "invalid_request_error", 401: "invalid_request_error", 426: "upgrade_required",
               502: "server_error"},
}


def is_block_code(code: str) -> bool:
    """
    Description: Decide whether a string is a safe reason code to show the user.
    Parameters:
        code: a candidate such as "HARPO-BLOCK-UNKNOWN-TYPE".
    Output: True for HARPO-BLOCK- followed by upper-case letters and digits in hyphen-separated parts, at most 48
        characters, that does not spell any phrase Claude Code uses to start its own recovery. False otherwise.
    """
    if len(code) > _MAX_CODE_LENGTH or not _BLOCK_CODE.fullmatch(code):
        return False
    spaced = code.lower().replace("-", " ")
    return not any(phrase in spaced for phrase in _RECOVERY_PHRASES)


def block_message(code: str) -> str:
    """
    Description: Write the plain message a block shows the user.
    Parameters:
        code: a reason code for which is_block_code is true or that is listed in REASONS.
    Output: One sentence naming the code, a line of advice for the few codes that have one (a secret in provider-signed
        text: start a new session), then a pointer to `harpo doctor`. It holds no text taken from the request. Raises
        ValueError for any other code.
    """
    if code not in REASONS and not is_block_code(code):
        raise ValueError("not a reason code")
    if code == UPSTREAM_FAILED:
        return f"Harpo could not get a reply from the provider (reason {code})."
    return (f"Harpo blocked this request before it left your machine (reason {code}). {_ADVICE.get(code, '')}"
            "Run `harpo doctor` for details.")


def error_response(provider: str, code: str) -> tuple[int, bytes]:
    """
    Description: Build the status and body of an answer the gate gives itself.
    Parameters:
        provider: "anthropic" for the {"type": "error", "error": {...}} shape, anything else for the OpenAI shape.
        code: a reason code; see block_message.
    Output: (HTTP status, JSON body bytes). The status is REASONS[code] (400 for a scan-step code). Raises ValueError for
        a code that is not a reason code.
    """
    message = block_message(code)
    status = REASONS.get(code, 400)
    family = "anthropic" if provider == "anthropic" else "openai"
    kind = _KINDS[family][status]
    if family == "anthropic":
        body = {"type": "error", "error": {"type": kind, "message": message}}
    else:
        body = {"error": {"message": message, "type": kind, "param": None, "code": None}}
    return status, json.dumps(body, separators=(",", ":")).encode("ascii")
