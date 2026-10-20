"""FR-INSTALL-04: request bodies arrive as identity, gzip or zstd; anything else is blocked.

Unit tests of the decode/encode step. A zstd test uses the `zstandard` package when installed; a fake module shaped
like Python 3.14's compression.zstd covers the loader for that interpreter.
"""
from __future__ import annotations

import gzip
import importlib
import os
import time
import types

import pytest

from Harpocrates.gate import body_encoding
from Harpocrates.gate.body_encoding import Decoded, Zstd, decode_body, encode_body, load_zstd

zstandard = pytest.importorskip("zstandard")
CAP = 1 << 20
PLAIN = b'{"model":"m","messages":[{"role":"user","content":"hello"}]}'


@pytest.fixture
def zstd() -> Zstd:
    codec = load_zstd()
    assert codec is not None
    return codec


# --- identity ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("header", [[], ["identity"], ["IDENTITY"], [" identity "]])
def test_fr_install_04_identity_passes_unchanged(header: list[str]) -> None:
    assert decode_body(PLAIN, header, CAP, None) == Decoded("ok", PLAIN, "identity")


def test_fr_install_04_identity_over_the_cap_is_too_large() -> None:
    assert decode_body(b"x" * (CAP + 1), [], CAP, None).status == "too-large"
    assert decode_body(b"x" * CAP, [], CAP, None).status == "ok"


# --- gzip -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("header", ["gzip", "GZIP", " GZip "])
def test_fr_install_04_gzip_is_decoded(header: str) -> None:
    assert decode_body(gzip.compress(PLAIN), [header], CAP, None) == Decoded("ok", PLAIN, "gzip")


def test_fr_install_04_gzip_at_the_cap_is_ok_and_one_over_is_too_large() -> None:
    assert decode_body(gzip.compress(b"a" * CAP), ["gzip"], CAP, None).status == "ok"
    assert decode_body(gzip.compress(b"a" * (CAP + 1)), ["gzip"], CAP, None).status == "too-large"


def test_fr_install_04_gzip_bomb_is_too_large_and_never_fully_expanded() -> None:
    bomb = gzip.compress(b"\0" * (200 * 1024 * 1024), compresslevel=9)
    assert len(bomb) < 1024 * 1024
    started = time.monotonic()
    result = decode_body(bomb, ["gzip"], CAP, None)
    assert result == Decoded("too-large", b"", "gzip")
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("make", [
    lambda: os.urandom(64),
    lambda: gzip.compress(PLAIN)[:-6],
    lambda: gzip.compress(PLAIN) + b"trailing",
    lambda: gzip.compress(PLAIN) + gzip.compress(b"second member"),
    lambda: b"",
    lambda: b"\x1f\x8b",
])
def test_fr_install_04_gzip_corrupt_truncated_or_padded_is_corrupt(make) -> None:
    assert decode_body(make(), ["gzip"], CAP, None) == Decoded("corrupt", b"", "gzip")


# --- everything else --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("header", [["br"], ["deflate"], ["x-gzip"], ["compress"], ["gzip, br"], ["identity, gzip"],
                                    ["zstd, gzip"], [""], ["gzip", "gzip"], ["gzip;q=1"], ["unknown"]])
def test_fr_install_04_unknown_encoding_is_unsupported(header: list[str]) -> None:
    result = decode_body(gzip.compress(PLAIN), header, CAP, None)
    assert result.status == "unsupported" and result.body == b""


# --- zstd -------------------------------------------------------------------------------------------------------

def test_fr_install_04_zstd_is_decoded_when_a_module_is_available(zstd: Zstd) -> None:
    raw = zstandard.ZstdCompressor().compress(PLAIN)
    assert decode_body(raw, ["zstd"], CAP, zstd) == Decoded("ok", PLAIN, "zstd")
    streamed = zstandard.ZstdCompressor().compressobj()
    raw = streamed.compress(PLAIN) + streamed.flush()  # a frame with no declared size, as a streaming client sends
    assert decode_body(raw, ["zstd"], CAP, zstd) == Decoded("ok", PLAIN, "zstd")


def test_fr_install_04_zstd_without_a_module_is_blocked() -> None:
    raw = zstandard.ZstdCompressor().compress(PLAIN)
    assert decode_body(raw, ["zstd"], CAP, None) == Decoded("zstd-unavailable", b"", "zstd")


def test_fr_install_04_zstd_at_the_cap_is_ok_and_one_over_is_too_large(zstd: Zstd) -> None:
    c = zstandard.ZstdCompressor()
    assert decode_body(c.compress(b"a" * CAP), ["zstd"], CAP, zstd).status == "ok"
    assert decode_body(c.compress(b"a" * (CAP + 1)), ["zstd"], CAP, zstd).status == "too-large"


def test_fr_install_04_zstd_bomb_is_too_large_and_never_fully_expanded(zstd: Zstd) -> None:
    stream = zstandard.ZstdCompressor().compressobj()
    bomb = stream.compress(b"\0" * (200 * 1024 * 1024)) + stream.flush()
    known = zstandard.ZstdCompressor().compress(b"\0" * (200 * 1024 * 1024))
    started = time.monotonic()
    for raw in (bomb, known):
        assert len(raw) < 1024 * 1024
        assert decode_body(raw, ["zstd"], CAP, zstd) == Decoded("too-large", b"", "zstd")
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("make", [
    lambda: os.urandom(64),
    lambda: zstandard.ZstdCompressor().compress(PLAIN)[:-5],
    lambda: zstandard.ZstdCompressor().compress(PLAIN) + b"xx",
    lambda: zstandard.ZstdCompressor().compress(PLAIN) * 2,
    lambda: b"",
])
def test_fr_install_04_zstd_corrupt_truncated_or_padded_is_corrupt(make, zstd: Zstd) -> None:
    assert decode_body(make(), ["zstd"], CAP, zstd) == Decoded("corrupt", b"", "zstd")


# --- encode: back to the encoding the body arrived in -----------------------------------------------------------

def test_fr_install_04_encode_restores_the_arrival_encoding(zstd: Zstd) -> None:
    assert encode_body(PLAIN, "identity", None) == PLAIN
    assert decode_body(encode_body(PLAIN, "gzip", None), ["gzip"], CAP, None).body == PLAIN
    assert decode_body(encode_body(PLAIN, "zstd", zstd), ["zstd"], CAP, zstd).body == PLAIN


def test_fr_install_04_gzip_encoding_is_deterministic() -> None:
    assert encode_body(PLAIN, "gzip", None) == encode_body(PLAIN, "gzip", None)


def test_fr_install_04_encode_zstd_without_a_module_or_an_unknown_encoding_raises() -> None:
    with pytest.raises(ValueError):
        encode_body(PLAIN, "zstd", None)
    with pytest.raises(ValueError):
        encode_body(PLAIN, "br", None)


# --- finding a zstd module --------------------------------------------------------------------------------------

def test_fr_install_04_load_zstd_returns_none_when_no_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(name: str, package=None):
        raise ImportError(name)

    monkeypatch.setattr(body_encoding.importlib, "import_module", refuse)
    assert load_zstd() is None


def test_fr_install_04_load_zstd_uses_the_python_3_14_module_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDecompressor:
        """Shaped like compression.zstd.ZstdDecompressor: decompress(data), eof, unused_data."""

        def __init__(self) -> None:
            self._inner = zstandard.ZstdDecompressor().decompressobj()

        def decompress(self, data: bytes, max_length: int = -1) -> bytes:
            return self._inner.decompress(data)

        @property
        def eof(self) -> bool:
            return self._inner.eof

        @property
        def unused_data(self) -> bytes:
            return self._inner.unused_data

    fake = types.SimpleNamespace(ZstdDecompressor=FakeDecompressor,
                                 compress=lambda data: zstandard.ZstdCompressor().compress(data))
    real_import = importlib.import_module
    monkeypatch.setattr(body_encoding.importlib, "import_module",
                        lambda name, package=None: fake if name == "compression.zstd" else real_import(name, package))
    codec = load_zstd()
    assert codec is not None
    raw = encode_body(PLAIN, "zstd", codec)
    assert decode_body(raw, ["zstd"], CAP, codec) == Decoded("ok", PLAIN, "zstd")
    assert decode_body(raw[:-3], ["zstd"], CAP, codec).status == "corrupt"
