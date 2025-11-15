from Harpocrates.scanner.entropy import shannon_entropy, looks_like_secret

def test_low_entropy():
    assert shannon_entropy("aaaaaa") < 1.0
    
def test_high_entropy():
    assert shannon_entropy("A7f$kL9#8!") > 3.0
    
def test_secret_detection():
    assert looks_like_secret("A7f$kL9#8!") is True
