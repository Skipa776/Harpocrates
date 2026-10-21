"""FR-GATE-01 (G3 spec, "Body"): read a request body by Content-Length or chunked, never past the cap, and say
plainly when the framing is wrong or the body is cut short. Unit tests over in-memory streams."""
from __future__ import annotations

import io

import pytest

from Harpocrates.gate.body import BodyRead, read_body

CAP = 1024


def read(raw: bytes, lengths=(), encodings=(), cap: int = CAP) -> tuple[BodyRead, io.BytesIO]:
    stream = io.BytesIO(raw)
    return read_body(stream, list(lengths), list(encodings), cap), stream


def test_fr_gate_01_no_framing_means_an_empty_body_and_nothing_is_read() -> None:
    result, stream = read(b"left over")
    assert result == BodyRead("ok", b"") and stream.tell() == 0


def test_fr_gate_01_content_length_body_is_read_exactly() -> None:
    result, stream = read(b"helloEXTRA", lengths=["5"])
    assert result == BodyRead("ok", b"hello") and stream.tell() == 5


def test_fr_gate_01_content_length_zero_is_an_empty_ok_body() -> None:
    assert read(b"", lengths=["0"])[0] == BodyRead("ok", b"")


def test_fr_gate_01_content_length_padding_whitespace_is_ignored() -> None:
    assert read(b"hi", lengths=[" 2\t"])[0] == BodyRead("ok", b"hi")


def test_fr_gate_01_content_length_body_at_the_cap_is_ok_and_one_over_is_too_large() -> None:
    assert read(b"x" * CAP, lengths=[str(CAP)])[0].status == "ok"
    result, stream = read(b"x" * (CAP + 1), lengths=[str(CAP + 1)])
    assert result == BodyRead("too-large", b"") and stream.tell() == 0  # refused from the header alone


def test_fr_gate_01_huge_content_length_is_too_large_without_parsing_a_huge_number() -> None:
    assert read(b"", lengths=["9" * 5000])[0].status == "too-large"
    assert read(b"", lengths=["99999999999999999999"])[0].status == "too-large"


def test_fr_gate_01_short_content_length_body_is_truncated() -> None:
    assert read(b"hello", lengths=["10"])[0] == BodyRead("truncated", b"")


@pytest.mark.parametrize("value", ["abc", "-1", "+5", "5_0", "٣", "1e3", "", "0x5", "5 5", "5.0"])
def test_fr_gate_01_bad_content_length_is_bad_framing(value: str) -> None:
    assert read(b"hello", lengths=[value])[0] == BodyRead("bad-framing", b"")


def test_fr_gate_01_two_content_lengths_are_bad_framing_even_when_equal() -> None:
    assert read(b"hello", lengths=["5", "5"])[0].status == "bad-framing"


def test_fr_gate_01_content_length_with_transfer_encoding_is_bad_framing() -> None:
    assert read(b"5\r\nhello\r\n0\r\n\r\n", lengths=["5"], encodings=["chunked"])[0].status == "bad-framing"


@pytest.mark.parametrize("encodings", [["gzip, chunked"], ["identity"], ["chunked", "chunked"], ["chunked, gzip"],
                                       ["gzip"], [""]])
def test_fr_gate_01_only_a_single_chunked_transfer_encoding_is_accepted(encodings: list[str]) -> None:
    assert read(b"0\r\n\r\n", encodings=encodings)[0].status == "bad-framing"


@pytest.mark.parametrize("raw,expected", [
    (b"5\r\nhello\r\n0\r\n\r\n", b"hello"),
    (b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n", b"abcde"),
    (b"5;ext=1;other\r\nhello\r\n0\r\n\r\n", b"hello"),
    (b"A\r\n0123456789\r\n0\r\n\r\n", b"0123456789"),
    (b"a\r\n0123456789\r\n0\r\n\r\n", b"0123456789"),
    (b"5\r\nhello\r\n0\r\nX-Trailer: 1\r\nY: 2\r\n\r\n", b"hello"),
    (b"0\r\n\r\n", b""),
    (b"4\r\n\r\n\r\r\r\n0\r\n\r\n", b"\r\n\r\r"),  # data may hold CR and LF
])
def test_fr_gate_01_chunked_body_is_joined(raw: bytes, expected: bytes) -> None:
    assert read(raw, encodings=["chunked"])[0] == BodyRead("ok", expected)


def test_fr_gate_01_chunked_marker_is_case_insensitive() -> None:
    assert read(b"2\r\nhi\r\n0\r\n\r\n", encodings=["Chunked"])[0] == BodyRead("ok", b"hi")


def test_fr_gate_01_chunked_reading_stops_at_the_end_of_the_body() -> None:
    result, stream = read(b"2\r\nhi\r\n0\r\n\r\nNEXT REQUEST", encodings=["chunked"])
    assert result.status == "ok" and stream.read() == b"NEXT REQUEST"


@pytest.mark.parametrize("raw", [
    b"zz\r\nhello\r\n0\r\n\r\n", b"-1\r\nhello\r\n0\r\n\r\n", b"0x5\r\nhello\r\n0\r\n\r\n", b" 5\r\nhello\r\n0\r\n\r\n",
    b"\r\nhello\r\n0\r\n\r\n", b"11111111111111111\r\nx\r\n0\r\n\r\n", b"5\nhello\r\n0\r\n\r\n",
    b"5\r\nhelloXX0\r\n\r\n", b"5\r\nhello\n0\r\n\r\n", b"5 \r\nhello\r\n0\r\n\r\n",
])
def test_fr_gate_01_malformed_chunks_are_bad_framing(raw: bytes) -> None:
    assert read(raw, encodings=["chunked"])[0] == BodyRead("bad-framing", b"")


def test_fr_gate_01_a_chunk_size_line_without_an_end_is_bad_framing() -> None:
    assert read(b"5;" + b"x" * 10000, encodings=["chunked"])[0].status == "bad-framing"


def test_fr_gate_01_a_flood_of_trailers_is_bad_framing() -> None:
    raw = b"0\r\n" + b"X: 1\r\n" * 200 + b"\r\n"
    assert read(raw, encodings=["chunked"])[0].status == "bad-framing"


@pytest.mark.parametrize("raw", [b"", b"5\r\nhel", b"5\r\nhello", b"5\r\nhello\r\n", b"5\r\nhello\r\n0\r\n",
                                 b"5\r\nhello\r\n0\r\nX: 1\r\n"])
def test_fr_gate_01_a_chunked_body_cut_short_is_truncated(raw: bytes) -> None:
    assert read(raw, encodings=["chunked"])[0] == BodyRead("truncated", b"")


def test_fr_gate_01_chunked_body_over_the_cap_is_too_large() -> None:
    assert read(b"6\r\nabcdef\r\n6\r\nghijkl\r\n0\r\n\r\n", encodings=["chunked"], cap=10)[0].status == "too-large"
    result, stream = read(b"ff\r\n" + b"x" * 255 + b"\r\n0\r\n\r\n", encodings=["chunked"], cap=10)
    assert result.status == "too-large" and stream.tell() == 4  # refused from the size line, data not read
    assert read(b"a\r\n0123456789\r\n0\r\n\r\n", encodings=["chunked"], cap=10)[0].status == "ok"


class Stalling(io.BytesIO):
    """A stream that times out after its bytes run out, like a client that stops sending."""

    def read(self, size: int = -1) -> bytes:
        data = super().read(size)
        if not data and size != 0:
            raise TimeoutError("timed out")
        return data

    def readline(self, size: int = -1) -> bytes:
        data = super().readline(size)
        if not data:
            raise TimeoutError("timed out")
        return data


def test_fr_gate_01_a_client_that_stalls_mid_body_is_truncated() -> None:
    assert read_body(Stalling(b"hel"), ["10"], [], CAP) == BodyRead("truncated", b"")
    assert read_body(Stalling(b"5\r\nhel"), [], ["chunked"], CAP) == BodyRead("truncated", b"")
    assert read_body(Stalling(b"5\r\nhello\r\n"), [], ["chunked"], CAP) == BodyRead("truncated", b"")


def test_fr_gate_01_large_bodies_are_read_in_pieces() -> None:
    body = b"y" * (3 * 1024 * 1024 + 7)
    assert read(body, lengths=[str(len(body))], cap=len(body))[0] == BodyRead("ok", body)
