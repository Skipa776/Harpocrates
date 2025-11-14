from entropy import shannon_entropy

def detect_entropy_secrets(text: str, threshold: float = 4.0):
    entropy = shannon_entropy(text)
    return entropy >= threshold, entropy