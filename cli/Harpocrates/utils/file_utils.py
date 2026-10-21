"""Text-file reading for the scanners: a binary probe, and lines read lazily in bounded pieces (FR-CORE-05)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

MEMORY_CAP = 64 * 1024 * 1024  # bytes one file's scan may hold (Josh, Oct 21, 2026)
PIECE_OVERLAP = 64 * 1024  # characters; longer than any token the regex and entropy tiers match


def _looks_binary(sample: bytes) -> bool:
    """
    Description: Inspect a byte sample for binary content before attempting text scanning.
    Parameters:
        sample: A sample of the file content.
    Output: True for NUL bytes or over 30% nontext bytes after failed UTF-8 decoding; False for
    empty, valid UTF-8, or a truncated final UTF-8 character.
    Heuristic check for binary files. bool: True if the file is binary, False otherwise.
    """
    if not sample:
        return False

    if b"\x00" in sample:
        return True

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as error:
        # A probe can end midway through a valid multi-byte character.
        if error.end == len(sample) and error.reason == "unexpected end of data":
            return False
    else:
        return False

    nontext = 0
    for byte in sample:
        if byte in b"\t\n\r\f\b":
            continue

        if byte < 32 or byte > 126:
            nontext += 1

    return nontext / len(sample) > 0.3

def piece_chars(memory_cap: int) -> int:
    """
    Description: The longest piece of a line one scan holds, for a memory cap (FR-CORE-05).
    Parameters:
        memory_cap: the most bytes one file's scan may hold.
    Output: A quarter of the cap, and at least twice PIECE_OVERLAP.
    """
    return max(memory_cap // 4, 2 * PIECE_OVERLAP)


def iter_line_pieces(path: str | Path, memory_cap: int = MEMORY_CAP) -> Iterator[tuple[int, int, str]]:
    """
    Description: Read a text file's lines lazily, cutting a line longer than piece_chars(memory_cap) into pieces that
        overlap by PIECE_OVERLAP characters, so a scan of any size holds one piece at a time (FR-CORE-05).
    Parameters:
        path: the text file.
        memory_cap: the most bytes one file's scan may hold (default MEMORY_CAP).
    Output: (line number from 1, offset of the piece in its line, text without the newline) for each piece; a short
        line is one piece at offset 0. Nothing for a binary file (a 1024-byte probe). UTF-8, with undecodable bytes
        dropped. Raises OSError.
    """
    path = Path(path)
    with path.open("rb") as f:
        if _looks_binary(f.read(1024)):
            return
    size = piece_chars(memory_cap)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lineno, read, carry = 1, 0, ""  # read: characters of this line consumed so far
        while text := f.readline(size - len(carry)):
            ended = text.endswith("\n")
            piece = carry + (text[:-1] if ended else text)
            yield lineno, read - len(carry), piece
            if ended:
                lineno, read, carry = lineno + 1, 0, ""
            else:
                read += len(text)
                carry = piece[-PIECE_OVERLAP:]


def count_lines(path: str | Path, memory_cap: int = MEMORY_CAP) -> int:
    """
    Description: Count a text file's lines without holding a whole line (FR-CORE-05).
    Parameters:
        path: the text file.
        memory_cap: the most bytes the count may hold.
    Output: The line count; 0 for a binary file. Raises OSError.
    """
    return sum(1 for _, offset, _ in iter_line_pieces(path, memory_cap) if offset == 0)
