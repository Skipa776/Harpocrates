from math import log2

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    
    freq = {c: s.count(c) for c in set(s)}
    return -sum((count/len(s)) * log2(count/len(s)) for count in freq.values())