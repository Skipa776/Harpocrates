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
    
    def __init__(self, scanner: Optional[list[BaseScanner]] = None) -> None:
        """
        Configure the Detector with a list of scanners to run.
        
        Parameters:
            scanner (Optional[list[BaseScanner]]): Ordered list of scanner instances to use for detection.
                If omitted or None, defaults to a list containing a RegexScanner configured with
                SIGNATURES and an EntropyScanner.
        """
        self.scanners: list[BaseScanner] = scanner or [
            RegexScanner(signatures=SIGNATURES),
            EntropyScanner(),
        ]
    
    def scan_text(self, text: str, file_path: Optional[str] = None) -> List[Finding]:
        """
        Run configured scanners over a text blob and collect their findings.
        
        Parameters:
            text (str): The full text to scan.
            file_path (Optional[str]): Optional file path used to populate scanner context (e.g., for reporting line/file information).
        
        Returns:
            List[Finding]: A list of findings produced by the configured scanners for the provided text.
        """
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
        Scan the text contents of a file and return detected findings.
        
        Parameters:
        	path (str | Path): Path to the file to read and scan.
        	max_bytes (Optional[int]): Maximum number of bytes to read from the file; `None` to read the entire file.
        
        Returns:
        	findings (List[Finding]): Detected findings for the file. The returned findings include the file path string as the `file_path` context.
        """
        path_obj = Path(path)
        file_name = str(path_obj)
        
        lines: list[str] = []
        for _, line in iter_text_lines(path_obj, max_bytes=max_bytes):
            lines.append(line.rstrip("\n"))
        text = "\n".join(lines)
        return self.scan_text(text, file_path=file_name)
    
    def _post_process(self, findings: List[Finding], full_text: str) -> None:
        """
        Adjusts entropy-based findings by increasing their confidence when the finding's raw text appears on the right-hand side of a simple assignment on the same line.
        
        Only findings produced by the EntropyScanner that include a line_number are considered. For each such finding, if the corresponding source line matches the detector's simple assignment pattern and the finding's raw_text is contained within the captured value, the finding's confidence is increased by 0.2 via its `enhance_confidence` method.
        
        Parameters:
            findings (List[Finding]): Collected findings to post-process; modified in place.
            full_text (str): Full file text used to locate source lines referenced by findings.
        """
        lines = full_text.splitlines()
        
        for f in findings:
            if f.scanner_name != "EntropyScanner":
                continue
            if f.line_number is None:
                continue
            
            idx = f.line_number -1
            if not( 0<= idx < len(lines)):
                continue
            line = lines[idx]
            if self._looks_like_assignment(line, f.raw_text):
                f.enhance_confidence(0.2)

    def _looks_like_assignment(self, line: str, raw_text: str) -> bool:
        """
        Determine whether a source line is a simple `name = value` assignment whose value contains the provided raw text.
        
        Parameters:
            line (str): A single line of text to test against the assignment pattern.
            raw_text (str): The substring to search for inside the captured assignment value.
        
        Returns:
            True if `line` matches the assignment pattern and `raw_text` is contained within the captured value, False otherwise.
        """
        match = self.ASSIGNMENT_PATTERN.match(line)
        if not match:
            return False
        value = match.group("value")
        return raw_text in value

def detect_text(text: str, threshold: float = 4.0) -> List[Finding]:
    """
    Scan the given text for findings using the project's regex and entropy scanners.
    
    This is a backwards-compatible wrapper that constructs a Detector with a RegexScanner and an EntropyScanner and runs a scan over the provided text.
    
    Parameters:
        threshold (float): Entropy threshold passed to EntropyScanner to control sensitivity; higher values make the entropy check stricter.
    
    Returns:
        List[Finding]: A list of detected findings found in the text.
    """
    detector = Detector(
        scanner=[
            RegexScanner(signatures=SIGNATURES),
            EntropyScanner(entropy_threshold=threshold),
        ]
    )
    return detector.scan_text(text, file_path=None)

def detect_file(path: str | Path, threshold: float = 4.0, max_bytes: Optional[int] = None) -> List[Finding]:
    """
    Scan a file for findings using regex signatures and entropy-based detection.
    
    Parameters:
        path (str | Path): Path to the file to scan.
        threshold (float): Entropy threshold applied by the entropy-based scanner.
        max_bytes (Optional[int]): Maximum number of bytes to read from the file; `None` means no limit.
    
    Returns:
        List[Finding]: Findings discovered in the file.
    """
    detector = Detector(
        scanner=[
            RegexScanner(signatures=SIGNATURES),
            EntropyScanner(entropy_threshold=threshold),
        ]
    )
    return detector.scan_file(path, max_bytes=max_bytes)