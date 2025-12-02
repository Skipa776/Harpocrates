from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from Harpocrates.scanner.base import BaseScanner
from Harpocrates.scanner.models import Finding
from Harpocrates.scanner.entropy import EntropyScanner
from Harpocrates.scanner.regex_signatures import RegexScanner, SIGNATURES
from Harpocrates.utils.file_loader import iter_text_lines


class Detector:
    '''
    Orchestrates all scanners and applies cross-scanner post_processing.
    '''
    ASSIGNMENT_PATTERN = re.compile(
        r"""
        ^\s*
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        \s*=\s*
        (?P<value>["'].*["']|[A-Za-z0-9+/_=.-]+)
        """,
        re.VERBOSE,
    )
    
    def __init__(self, scanners: Optional[list[BaseScanner]] = None) -> None:
        self.scanners: list[BaseScanner] = scanners or [
            RegexScanner(signatures=SIGNATURES),
            EntropyScanner(),
        ]
    
    def scan_text(self, text: str, file_path: Optional[str] = None) -> List[Finding]:
        '''
        Scan a text blob and return Findings.
        '''
        context = {'file_path': file_path}
        findings: List[Finding] = []
        for scanner in self.scanners:
            findings.extend(scanner.scan(text, context))
            
        self._post_process(findings, text)
        return findings
    
    def scan_file(
        self,
        path: str | Path,
        max_bytes: Optional[int] = None,
    ) -> List[Finding]:
        """
        Scan a file and return Findings.
        """
        path_obj = Path(path)
        file_name = str(path_obj)
        
        lines: list[str] = []
        for _, line in iter_text_lines(path_obj, max_bytes=max_bytes):
            lines.append(line)
        text = "\n".join(lines)
        return self.scan_text(text, file_path=file_name)
    
    def _post_process(self, findings: List[Finding], full_text: str) -> None:
        lines = full_text.splitlines()
        
        for f in findings:
            if f.scanner_name != "EntropyScanner":
                continue
            idx = f.line_number - 1
            if not (0 <= idx < len(lines)):
                continue
            line = lines[idx]
            if self._looks_like_assignment(line, f.raw_text):
                f.enhance_confidence(0.2)

    def _looks_like_assignment(self, line: str, raw_text: str) -> bool:
        '''
        Heuristic: does this line look like name = value and contain raw text?
        '''
        match = self.ASSIGNMENT_PATTERN.match(line)
        if not match:
            return False
        value = match.group("value")
        return raw_text in value

def detect_text(text: str, threshold: float = 4.0) -> List[Finding]:
    '''
    Backwards-compatible wrapper around Detector.scan_text.

    `threshold` can be plumbed into EntropyScanner in the future;
    for now it's unused to keep the signature stable.
    '''
    detector = Detector(
        scanners=[
            RegexScanner(signatures=SIGNATURES),
            EntropyScanner(entropy_threshold=threshold),
        ]
    )
    return detector.scan_text(text, file_path=None)

def detect_file(path: str | Path, threshold: float = 4.0, max_bytes: Optional[int] = None) -> List[Finding]:
    """
    Backwards-compatible wrapper around Detector.scan_file.
    """
    detector = Detector(
        scanners=[
            RegexScanner(signatures=SIGNATURES),
            EntropyScanner(entropy_threshold=threshold),
        ]
    )
    return detector.scan_file(path, max_bytes=max_bytes)