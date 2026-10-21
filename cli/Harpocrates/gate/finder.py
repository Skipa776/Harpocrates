"""Find secrets in text and write their placeholders (FR-GATE-04, FR-GATE-05, SEC-02).

One Finder lives for one gate session. It owns the session key (the Redactor's, random and in memory only) and the cache
of findings. A leaf's spans depend only on its text (the detector is the read tool's, at the gate threshold), so a leaf
seen before is answered from the cache, and the placeholder of a value depends only on the value and the session key.
"""
from __future__ import annotations

from typing import Callable, Optional

from Harpocrates.core.detector import _apply_ml_verification, _collect_text_findings
from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.gate.spancache import Span, SpanCache, cache_key
from Harpocrates.read import Redactor, _gate_verifier

Detector = Callable[[str], list]


def strict_findings(text: str) -> list[Finding]:
    """
    Description: Detect secrets in text the way the read tool does (signature regexes, then entropy candidates verified
        by the model at the gate threshold from model_config.json: recall first), except that a verifier error is an error.
    Parameters:
        text: the text to scan.
    Output: Regex findings followed by the candidates the model keeps. Raises whatever the detector or the model raises
        (the read tool would redact every candidate instead); the scan step turns that into a block.
    """
    findings = _collect_text_findings(text)
    regex = [f for f in findings if f.evidence == EvidenceType.REGEX]
    candidates = [f for f in findings if f.evidence != EvidenceType.REGEX]
    if not candidates:
        return regex
    return regex + _apply_ml_verification(candidates, text, _gate_verifier(), ml_threshold=0.0)


def _occurrences(text: str, values: dict) -> tuple[Span, ...]:
    """
    Description: Place every occurrence of every secret value in a text.
    Parameters:
        text: the text.
        values: {value: secret type}, longest value first.
    Output: Disjoint spans in order. A longer value is placed first and swallows a shorter one inside it; an occurrence that
        overlaps a span already placed is skipped.
    """
    placed: list[Span] = []
    for value, kind in values.items():
        at = text.find(value)
        while at >= 0:
            end = at + len(value)
            if not any(at < s.end and s.start < end for s in placed):
                placed.append(Span(at, end, kind))
            at = text.find(value, end)
    return tuple(sorted(placed))


class Finder:
    """
    Description: Finds secrets and rewrites them to placeholders for one gate session.
    Parameters:
        detector: a function text -> findings, or None for strict_findings.
        cache: the cache of findings, or None for a new one with the specified bounds.
        redactor: the Redactor whose random in-memory key makes the placeholders, or None for a new one (a new gate has a
            new key, SEC-02).
    Output: An object with spans, redact and has_secret; shares nothing between instances unless handed the same parts.
    """

    def __init__(self, detector: Optional[Detector] = None, cache: Optional[SpanCache] = None,
                 redactor: Optional[Redactor] = None) -> None:
        """
        Description: Keep the parts; make the missing ones.
        Parameters:
            detector: the detector, or None for strict_findings.
            cache: the cache, or None for a new SpanCache.
            redactor: the placeholder maker, or None for a new Redactor.
        Output: None.
        """
        self._detect = detector or strict_findings
        self.cache = cache if cache is not None else SpanCache()
        self.redactor = redactor if redactor is not None else Redactor()

    def spans(self, text: str, field_class: str) -> tuple[Span, ...]:
        """
        Description: Find the secrets in a leaf, from the cache when it has been seen.
        Parameters:
            text: the leaf text.
            field_class: "R", "S", "K", "H" or "B" (a base64 decoding, FR-GATE-11); part of the cache key.
        Output: The spans, possibly empty. Detector errors propagate and nothing is cached for them.
        """
        key = cache_key(field_class, text)
        known = self.cache.get(key)
        if known is not None:
            return known
        found = _occurrences(text, self.redactor.secret_values(text, findings=self._detect(text)))
        self.cache.put(key, found)
        return found

    def has_secret(self, text: str, field_class: str) -> bool:
        """
        Description: Tell whether a leaf holds a secret.
        Parameters:
            text: the leaf text.
            field_class: the field class.
        Output: True when spans() is not empty.
        """
        return bool(self.spans(text, field_class))

    def redact(self, text: str, field_class: str) -> Optional[str]:
        """
        Description: Replace each secret in a leaf with its placeholder <<HARPO:type:hash4>>.
        Parameters:
            text: the leaf text.
            field_class: the field class.
        Output: The new text, or None when the leaf holds no secret (so a clean leaf is never rewritten). The hash is the
            first four hex characters of HMAC-SHA256 of the value under the session key.
        """
        found = self.spans(text, field_class)
        if not found:
            return None
        pieces, cursor = [], 0
        for span in found:
            pieces += [text[cursor:span.start], self.redactor.placeholder(text[span.start:span.end], span.kind)]
            cursor = span.end
        pieces.append(text[cursor:])
        return "".join(pieces)
