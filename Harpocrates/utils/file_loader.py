import re
from Harpocrates.scanner.entropy import looks_like_secret, shannon_entropy

def scan_file_for_entropy(path: str, threshold: float = 4.0):
    findings = []
    
    with open(path, 'r', encoding = 'utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, 1):
            tokens = re.findall(r"[A-Za-z0-9+/=_-]{8,}", line)
            for token in tokens:
                if looks_like_secret(token, threshold):
                    findings.append({
                        "line": line_no,
                        "token": token,
                        "entropy": shannon_entropy(token)
                    })
    return findings