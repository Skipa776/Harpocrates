"""The reasons the scan step blocks a request, and the exception that carries one (FR-GATE-03).

Every code passes errors.is_block_code, so the user sees it and no text taken from the request can reach the message.
"""
from __future__ import annotations

PARSE = "HARPO-BLOCK-PARSE"  # not UTF-8, not JSON, a repeated key, nested too deep, or not an object
UNKNOWN_TYPE = "HARPO-BLOCK-UNKNOWN-TYPE"  # a block, item, tool or source type the parser spec does not list
PROVIDER_TEXT = "HARPO-BLOCK-PROVIDER-TEXT"  # a secret in text only the provider can rewrite (FR-GATE-10)
KEY_SECRET = "HARPO-BLOCK-KEY-SECRET"  # a secret in an object key, which two keys could collapse into one placeholder
HEADER_SECRET = "HARPO-BLOCK-HEADER-SECRET"  # a secret in a header value, the path or the query: none can be rewritten
HISTORY = "HARPO-BLOCK-HISTORY"  # server-side history the gate cannot see (previous_response_id, conversation)
MEDIA_MALFORMED = "HARPO-BLOCK-MEDIA-MALFORMED"  # a media field that is not strict base64 to its end (FR-GATE-07)
MEDIA_SECRET = "HARPO-BLOCK-MEDIA-SECRET"  # a secret in a text file sent as base64, which is not rewritten in v1
ENCODED_SECRET = "HARPO-BLOCK-ENCODED-SECRET"  # a secret in base64 inside a text field (FR-GATE-11)
ENCODED_TOO_LARGE = "HARPO-BLOCK-ENCODED-TOO-LARGE"  # more base64 in text than one request may decode (FR-GATE-11)
JSON_ARGS = "HARPO-BLOCK-JSON-ARGS"  # tool-call arguments that parsed before redaction but not after


class RefusalError(Exception):
    """
    Description: The scan step decided to block the request.
    Parameters:
        code: one of the codes in this module.
    Output: An exception with .code; its message is the code and nothing from the request.
    """

    def __init__(self, code: str) -> None:
        """
        Description: Keep the code.
        Parameters:
            code: the reason code.
        Output: None.
        """
        super().__init__(code)
        self.code = code
