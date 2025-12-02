from __future__ import annotations

import re
from re import Pattern
from typing import Dict, List, Any

from Harpocrates.scanner.base import BaseScanner
from Harpocrates.scanner.models import Finding

# Base regex signatures for common secret formats
# Keep patterns PRECOMPILED for performance; keep them strict enough to reduce noise

SIGNATURES: Dict[str, Pattern] = {
    # --- AWS ---
    # Typical Access Key ID: "AKIA" followed by 16 uppercase alphanumeric characters
    "AWS_ACCESS_KEY_ID": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Secret Access Key: 40 base64-like characters
    # Replace your AWS secret entry with this:
"AWS_SECRET_ACCESS_KEY": re.compile(r"(?<![A-Za-z0-9/+])[A-Za-z0-9/+][A-Za-z0-9/+=]{39}(?![A-Za-z0-9/+=])"),

    
    # --- GitHub ---
    # Personal Access Token (classic). We'll start with the common 'ghp_' prefix.
    # (Fine-grained tokens have different prefixes; will add later)
    "GITHUB_TOKEN": re.compile(r"\bghp_[0-9a-zA-Z]{36}\b"),
    
    # --- GCP ---
    # API key often embedded in client_side code; starts with "AIza" and is 39 characters long.
    "GCP_API_KEY": re.compile(r"\bAIza[0-9A-Za-z-_]{35}\b"),
    
    # --- AZURE ---
    # API Key Vault URL; prescence often correlates with secrets nearby in config.
    "AZURE_KEY_VAULT_URL": re.compile(r"\bhttps://[a-z0-9-]+\.vault\.azure\.net", re.IGNORECASE),
    
    # --- JWT ---
    # Very loose match of header.payload.signature (URL-safe base64 segments separated by dots)
    "JWT": re.compile(r"\beyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]*\b"),
    
    # --- Slack --- 
    # Common Slack tokens (team/app/bot/etc.)
    "SLACK_TOKEN": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]+\b"),

    # --- Stripe ---
    # Live secret key (test keys use sk_test_)
    "STRIPE_SECRET_KEY": re.compile(r"\bsk_live_[0-9A-Za-z]{24}\b"),
    
    # --- OpenAI ---
    # API key format for OpenAI usually starts with "sk-" followed by 48 alphanumeric characters
    "OPENAI_API_KEY": re.compile(r"\bsk-[0-9a-zA-Z]{48}\b"),
}

__all__ = ["SIGNATURES"]

class RegexScanner(BaseScanner):
    '''
    Scanner for regex signatures.
    '''
    
    def __init__(self, signatures: Dict[str, Pattern], base_confidence: float = 0.9) -> None:
        """
        Initialize the RegexScanner with a set of regex signatures and a baseline confidence.
        
        Parameters:
            signatures (Dict[str, Pattern]): Mapping of signature name to compiled regex pattern; if falsy, the module-level SIGNATURES mapping is used.
            base_confidence (float): Baseline confidence score applied to findings, typically between 0.0 and 1.0.
        """
        super().__init__(name="RegexScanner")
        self.signatures = signatures or SIGNATURES
        self.base_confidence = base_confidence
        
    def scan(self, content: str, context: Dict[str, Any]) -> List[Finding]:
        """
        Scan text content for known secret patterns and produce Findings for each match.
        
        Parameters:
            content (str): The text to scan; scanned line-by-line.
            context (Dict[str, Any]): Optional metadata used for findings (expects "file_path" for location).
        
        Returns:
            List[Finding]: A list of Finding objects for every regex match. Each Finding includes the signature name, file path (from context), line and column of the match, the raw matched token, a masked version of the token, a confidence score, and metadata with `type`, a 200-character snippet of the line, and `evidence` set to `"regex"`.
        """
        file_path = context.get("file_path")
        finding: List[Finding] = []
        
        for lineno, line in enumerate(content.splitlines(), start=1):
            for sig_name, pattern in self.signatures.items():
                for match in pattern.finditer(line):
                    token = match.group(0)
                    column = match.start()
                    finding.append(
                        Finding(
                            scanner_name=self.name,
                            signature_name=sig_name,
                            file_path=file_path,
                            line_number=lineno,
                            column=column,
                            raw_text=token,
                            masked_text=self._mask(token),
                            confidence_score=self.base_confidence,
                            metadata={
                                "type": sig_name,
                                "snippet": line.strip()[:200],
                                "evidence": "regex",
                            },
                        )
                    )
        return finding

    @staticmethod
    def _mask(value: str)-> str:
        """
        Mask a sensitive string by replacing its middle characters with asterisks.
        
        For strings longer than four characters, preserve the first two and last two characters and replace all characters in between with `*`. For strings of length four or less, replace every character with `*`.
        
        Parameters:
            value (str): The input string to mask.
        
        Returns:
            masked_value (str): The masked string.
        """
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]