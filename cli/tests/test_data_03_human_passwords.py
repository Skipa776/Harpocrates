"""DATA-03: generic passwords follow human patterns (words, digits, symbols), not uniform random."""

import random
import re

from Harpocrates.training.generators.secret_templates import generate_human_password

HUMAN_SHAPE = re.compile(r"^[A-Za-z]+[A-Za-z0-9_.@!#$%&*-]*$")


def test_data_03_passwords_are_word_based():
    random.seed(0)
    passwords = [generate_human_password() for _ in range(200)]
    assert all(HUMAN_SHAPE.match(p) for p in passwords), passwords[:5]
    # Every password starts with a dictionary-style word of 3+ letters.
    assert all(re.match(r"^[A-Za-z]{3,}", p) for p in passwords)
    # A bare word is too ambiguous to label a secret.
    assert all(re.search(r"[\d!@#$%&*._-]", p) for p in passwords)
    # Patterns vary: some have digits, some symbols, some both.
    assert any(re.search(r"\d", p) for p in passwords)
    assert any(re.search(r"[!@#$%&*._-]", p) for p in passwords)
    assert len(set(passwords)) > 150


def test_data_03_seeded_output_is_reproducible():
    random.seed(42)
    first = [generate_human_password() for _ in range(20)]
    random.seed(42)
    assert first == [generate_human_password() for _ in range(20)]
