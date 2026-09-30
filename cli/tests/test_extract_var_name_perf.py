"""extract_var_name must stay fast on minified lines (it was quadratic in word-character runs)."""

import time

from Harpocrates.core.classification import extract_var_name


def test_minified_line_with_huge_word_run_is_fast():
    blob = "A" * 150_000  # e.g. an inline base64 image in minified JS
    line = f'x="{blob}";var k="tok_9fQ2mZ7xL";'
    start = time.perf_counter()
    assert extract_var_name(line, "tok_9fQ2mZ7xL") == "k"
    assert time.perf_counter() - start < 0.5


def test_normal_lines_unchanged():
    assert extract_var_name('api_key = "abc123secretvalue"', "abc123secretvalue") == "api_key"
    assert extract_var_name('  DB_PASSWORD: "Wint3r#9"', "Wint3r#9") == "DB_PASSWORD"
    assert extract_var_name('connect("abc123secretvalue")', "abc123secretvalue") is None
    assert extract_var_name("TOKEN=abc123secretvalue", "TOKEN=abc123secretvalue") == "TOKEN"
