from math import log2
import string

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    
    freq = {c: s.count(c) for c in set(s)}
    return -sum((count/len(s)) * log2(count/len(s)) for count in freq.values())

def looks_like_secret(s: str, threshold: float = 4.0) -> bool:
    entropy = shannon_entropy(s)
    
    has_digits = any(c.isdigit() for c in s)
    has_letters = any(c.isalpha() for c in s)
    has_special = any(c in string.punctuation for c in s)

    return (
        entropy >= threshold and
        has_digits and
        has_letters and 
        len(s) >= 8
    )