"""The scan step (FR-GATE-02, FR-GATE-03, FR-GATE-04, FR-GATE-05, FR-GATE-10): parse a request, find the secrets in every
string the model will see, and forward it with only those strings rewritten.

Steps: scan the headers, path and query (decision 12: a header cannot be rewritten safely, so a finding there blocks;
headers that carry the user's own credentials are skipped, not scanned); parse the body strictly; walk it by its format's
tables; drop thinking blocks; scan each string (class R is rewritten, class S blocks on a finding, an object key blocks);
splice the new strings and the cuts into the original text. A request with nothing to change is forwarded as the same bytes.
"""
from __future__ import annotations

from typing import Iterator, Optional, Union
from urllib.parse import unquote, unquote_plus

from Harpocrates.gate import encoded, walk_anthropic, walk_responses
from Harpocrates.gate.blocks import (
    ENCODED_SECRET,
    ENCODED_TOO_LARGE,
    HEADER_SECRET,
    KEY_SECRET,
    MEDIA_SECRET,
    PARSE,
    PROVIDER_TEXT,
    RefusalError,
)
from Harpocrates.gate.finder import Finder
from Harpocrates.gate.jsontext import (
    BodyParseError,
    apply_edits,
    encode_literal,
    locate,
    parse_strict,
    redact_json_string,
)
from Harpocrates.gate.media import MediaNote, binary_kind, decode_strict, readings
from Harpocrates.gate.routes import Route
from Harpocrates.gate.scan import Block, Forward, RequestHead
from Harpocrates.gate.spancache import cache_key
from Harpocrates.gate.walk_common import Drop, KeyText, Leaf, Media

CREDENTIAL_WORDS = ("auth", "key", "token", "cookie", "secret")  # x-api-key, Authorization, x-codex-turn-state, set-cookie
CREDENTIAL_NAMES = frozenset({"chatgpt-account-id"})
CREDENTIAL_PREFIXES = ("anthropic-", "openai-")  # anthropic-version and -beta, OpenAI-Organization, -Project, -Beta


def is_credential_header(name: str) -> bool:
    """
    Description: Decide whether a header carries a credential or protocol setting the user sends on purpose.
    Parameters:
        name: the header name, any case.
    Output: True when the name contains auth, key, token, cookie or secret, is ChatGPT-Account-ID, or starts with
        anthropic- or openai-. Such a header is forwarded byte for byte and not scanned.
    """
    lowered = name.lower()
    return (lowered in CREDENTIAL_NAMES or lowered.startswith(CREDENTIAL_PREFIXES)
            or any(word in lowered for word in CREDENTIAL_WORDS))


def head_texts(head: RequestHead) -> Iterator[str]:
    """
    Description: List the strings of a request head that can hold a secret.
    Parameters:
        head: the headers, path and query.
    Output: Each non-credential header value, then the path and the query; a path or query that changes when its percent
        escapes are decoded is listed in both forms, so an encoded secret is found too.
    """
    for name, value in head.headers:
        if value and not is_credential_header(name):
            yield value
    for part, decode in ((head.path, unquote), (head.query, unquote_plus)):
        if part:
            yield part
            decoded = decode(part)
            if decoded != part:
                yield decoded


def scan_head(head: RequestHead, finder: Finder) -> None:
    """
    Description: Block a request whose head holds a secret.
    Parameters:
        head: the headers, path and query.
        finder: the session's finder.
    Output: None. Raises RefusalError(HEADER_SECRET) at the first finding; never names the header.
    """
    for text in head_texts(head):
        if finder.has_secret(text, "H"):
            raise RefusalError(HEADER_SECRET)


class RequestScanner:
    """
    Description: The real scan step of the gate: one instance per gate session, with its own finder (and so its own key
        and cache).
    Parameters:
        finder: the Finder to use, or None for a new one.
    Output: A callable (body, route, head) -> Forward | Block, usable as a ScanStep.
    """

    def __init__(self, finder: Optional[Finder] = None) -> None:
        """
        Description: Keep or make the finder.
        Parameters:
            finder: the Finder, or None for a new one (a new session key, an empty cache).
        Output: None.
        """
        self.finder = finder if finder is not None else Finder()

    def __call__(self, body: bytes, route: Route, head: RequestHead) -> Union[Forward, Block]:
        """
        Description: Decide what happens to one request.
        Parameters:
            body: the decoded request body (empty on a route that takes none).
            route: the route.
            head: the headers, path and query.
        Output: Forward with the bytes to send (the very same bytes when nothing needed to change), or Block with a reason
            code. A detector error propagates; run_scan turns it into a block.
        """
        try:
            return self._scan(body, route, head)
        except RefusalError as refusal:
            return Block(refusal.code)
        except BodyParseError:
            return Block(PARSE)

    def _scan(self, body: bytes, route: Route, head: RequestHead) -> Forward:
        """
        Description: Scan the head, then the body, and build what to forward.
        Parameters:
            body: the decoded request body.
            route: the route.
            head: the headers, path and query.
        Output: A Forward. Raises RefusalError or BodyParseError for a request that must be blocked.
        """
        scan_head(head, self.finder)
        if route.action == "pass":
            return Forward(body)
        text, tree = parse_strict(body)
        walk = walk_anthropic.walk if route.name.startswith("anthropic.") else walk_responses.walk
        replacements: dict[tuple, str] = {}
        drops: list[tuple] = []
        notes: list[MediaNote] = []
        budget = [encoded.MAX_RUN_CHARS]
        for op in walk(tree):
            if isinstance(op, Media):
                note = self._media(op)
                if note is not None:
                    notes.append(note)
            elif isinstance(op, Drop):
                drops.append(op.path)
            elif isinstance(op, KeyText):
                if self.finder.has_secret(op.text, "K"):
                    raise RefusalError(KEY_SECRET)
            elif op.cls == "S":
                if self.finder.has_secret(op.text, "S"):
                    raise RefusalError(PROVIDER_TEXT)
                self._encoded(op, budget)
            else:
                new = self._rewrite(op)
                self._encoded(op, budget)
                if new is not None:
                    replacements[op.path] = new
        if not replacements and not drops:
            return Forward(body, tuple(notes))
        found, cuts = locate(text, list(replacements), drops)
        edits = [(*found[path], encode_literal(new)) for path, new in replacements.items()]
        edits += [(start, end, "") for start, end in cuts]
        return Forward(apply_edits(text, edits).encode("utf-8"), tuple(notes))

    def _media(self, item: Media) -> Optional[MediaNote]:
        """
        Description: Check one media payload (FR-GATE-07).
        Parameters:
            item: the payload.
        Output: A MediaNote for a payload with a known binary signature, which passes unscanned; None for any other
            payload, which was read as text and holds no secret. Raises RefusalError(MEDIA_MALFORMED) for a payload that is
            not strict base64 and RefusalError(MEDIA_SECRET) for one that holds a secret (base64 is not rewritten in v1).
        """
        data = decode_strict(item.payload)
        kind = binary_kind(data)
        if kind is not None:
            return MediaNote(kind, len(data))
        if any(self.finder.has_secret(text, "S") for text in readings(data)):
            raise RefusalError(MEDIA_SECRET)
        return None

    def _encoded(self, leaf: Leaf, budget: list[int]) -> None:
        """
        Description: Look for a secret base64-encoded inside one class R or S string (FR-GATE-11).
        Parameters:
            leaf: the string and how to read it.
            budget: one-item list, the base64 characters this request may still decode; reduced by this leaf's runs.
        Output: None. Raises RefusalError(ENCODED_SECRET) when the text in any decoding of a base64 run in the leaf
            holds a secret, and RefusalError(ENCODED_TOO_LARGE) when the request's runs pass the budget. A leaf found
            clean is remembered by its hash (cache class "E", or "J" for a leaf read as JSON, which is searched more
            widely) and each decoding's findings under class "B", so a resent history is not decoded or scanned again;
            no text is kept.
        """
        key = cache_key("J" if leaf.as_json else "E", leaf.text)
        if self.finder.cache.get(key) is not None:
            return
        for text in encoded.strings_of(leaf.text, leaf.as_json):
            found = []
            for run in encoded.runs(text):  # the budget is checked as runs are found, before any decoding
                budget[0] -= len(run)
                if budget[0] < 0:
                    raise RefusalError(ENCODED_TOO_LARGE)
                found.append(run)
            for run in found:
                for decoded in encoded.texts_of_run(run):
                    if self.finder.has_secret(decoded, "B"):
                        raise RefusalError(ENCODED_SECRET)
        self.finder.cache.put(key, ())

    def _rewrite(self, leaf: Leaf) -> Optional[str]:
        """
        Description: Redact one class R string.
        Parameters:
            leaf: the string and how to read it.
        Output: The new text, or None when it holds no secret. A string that holds JSON is redacted leaf by leaf.
        """
        if leaf.as_json:
            return redact_json_string(leaf.text, self.finder)
        return self.finder.redact(leaf.text, "R")
