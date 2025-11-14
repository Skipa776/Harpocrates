from Harpocrates.scanner.entropy import shannon_entropy

def test_low_entropy():
    assert shannon_entropy("aaaaaa") < 1.0
    
def test_high_entropy():
    assert shannon_entropy("A7f$kL9#8!") > 3.0