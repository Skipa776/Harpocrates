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
    Heuristic check for binary files.
    Args:
        sample (bytes): A sample of the file content.

    Returns:
        bool: True if the file is binary, False otherwise.
    """
    if not sample:
        return False
    
    if b"\x00" in sample:
        return True
    
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
    Safely iterate over the lines of a text file.
    
    - Skips binary files (based on initial 1024-byte probe)
    - Uses UTF-8 encoding (errors ignored)
    - Respects max_bytes to avoid huge scans
    Args:
        path (str | Path): The path to the text file.
        max_bytes (int | None, optional): The maximum number of bytes to read. Defaults to None.

    Yields:
        (line_number, line_content)
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
            
            if max_bytes is not None and total_bytes>max_bytes:
                break

            yield lineno, line.rstrip("\n")
