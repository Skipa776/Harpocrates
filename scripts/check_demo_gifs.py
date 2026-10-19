#!/usr/bin/env python3
"""Fail when a README demo GIF is longer than 20 s or larger than 5 MB (DOC-DEMO-01).

    python scripts/check_demo_gifs.py docs/assets/demo-scan.gif docs/assets/demo-safe-read.gif

The length is the sum of the frame delays, read straight from the GIF, so no ffmpeg is needed.
"""
import sys
from pathlib import Path

MAX_SECONDS = 20.0
MAX_BYTES = 5 * 1024 * 1024


def _skip_sub_blocks(data, pos):
    while data[pos]:
        pos += data[pos] + 1
    return pos + 1


def duration_seconds(data):
    """
    Description: Add up the frame delays of a GIF by walking its blocks.
    Parameters:
        data: The bytes of a GIF file.
    Output: Seconds as a float; raises ValueError when the bytes are not a GIF.
    """
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("not a GIF")
    flags = data[10]
    pos = 13 + (3 * 2 ** ((flags & 7) + 1) if flags & 0x80 else 0)
    centiseconds = 0
    while pos < len(data) and data[pos] != 0x3B:
        if data[pos] == 0x21:  # extension
            if data[pos + 1] == 0xF9:  # graphic control: frame delay
                centiseconds += data[pos + 4] | (data[pos + 5] << 8)
            pos = _skip_sub_blocks(data, pos + 2)
        elif data[pos] == 0x2C:  # image descriptor
            local = data[pos + 9]
            pos += 10 + (3 * 2 ** ((local & 7) + 1) if local & 0x80 else 0) + 1  # + LZW minimum code size
            pos = _skip_sub_blocks(data, pos)
        else:
            raise ValueError(f"unexpected block 0x{data[pos]:02x}")
    return centiseconds / 100


def main(paths):
    """
    Description: Check each GIF against the length and size limits and print one line per file.
    Parameters:
        paths: GIF file paths from the command line.
    Output: Exit status 0 when every file is within both limits, 1 otherwise.
    """
    failed = False
    for name in paths:
        data = Path(name).read_bytes()
        seconds = duration_seconds(data)
        ok = seconds <= MAX_SECONDS and len(data) <= MAX_BYTES
        failed = failed or not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {seconds:.2f} s (max {MAX_SECONDS:g}), "
              f"{len(data) / 1024:.0f} KiB (max {MAX_BYTES // 1024})")
    return 1 if failed or not paths else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
