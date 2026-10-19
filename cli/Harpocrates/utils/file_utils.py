"""
This version:
- Detects binary files (skips them)
- Iterates lines lazily
- Support max_bytes cutoff
- UTF-8 with errors ignored
- Used by detect_file() and future CI scanning tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


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

def iter_text_lines(
    path: str | Path,
    max_bytes: int | None = None,
) -> Iterator[tuple[int, str]]:
    """
    Description: Read text lines lazily while skipping binary files and limiting scan size.
    Parameters:
        path: The path to the text file.
        max_bytes: The maximum number of bytes to read. Defaults to None.
    Output: Iterator of one-based line numbers and text without trailing newlines; opens and reads
    the file on iteration. Raises OSError for file access failures; decoding errors are ignored and
    lines crossing max_bytes are not yielded.
    Safely iterate over the lines of a text file.
    - Skips binary files (based on initial 1024-byte probe) - Uses UTF-8 encoding (errors ignored) -
    Respects max_bytes to avoid huge scans (line_number, line_content)
    """
    path = Path(path)

    with path.open("rb") as f:
        sample = f.read(1024)
        if _looks_binary(sample):
            return

    total_bytes = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            encoded_len = len(line.encode("utf-8", errors="ignore"))
            total_bytes += encoded_len

            if max_bytes is not None and total_bytes > max_bytes:
                break

            yield lineno, line.rstrip("\n")
