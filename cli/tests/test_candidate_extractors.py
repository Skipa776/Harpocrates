"""Candidate extraction: credential shapes the entropy stage never offered to the ML verifier.

These only add ML candidates; the ML stage decides. detect_text (no ML) must not
report them, so non-ML scans get no new noise.
"""

import pytest

from Harpocrates.core.detector import _collect_text_findings, detect_text
from Harpocrates.core.result import EvidenceType

# Fake values, assembled so this file holds no literal credential-shaped strings.
PW = "Wint3r" + "#Sales9"
HEX32 = "9f2c" + "4e1a" * 7


def _ml_tokens(line: str) -> set[str]:
    return {f.token for f in _collect_text_findings(line) if f.evidence == EvidenceType.ML}


@pytest.mark.parametrize("line, expected", [
    (f'url = "postgresql://app:{PW}@db.internal:5432/orders"',
     f"postgresql://app:{PW}@db.internal:5432/orders"),
    (f'conn = "Server=tcp:db,1433;Database=orders;User ID=admin;Password={PW};"', PW),
    (f'odbc = "Driver={{ODBC Driver 18 for SQL Server}};Server=db;Uid=sa;Pwd={PW};"', PW),
    (f'jdbc = "jdbc:mysql://db:3306/app?user=root&password={PW}"', PW),
    ('hook = "https://hooks.example.com/v1/notify?page=2&access_token=q8Lz0vT3nR7kW2pX9bY4"',
     "q8Lz0vT3nR7kW2pX9bY4"),
    (f'x = "{HEX32}"', HEX32),  # 32+ hex below the entropy gate, neutral name
    (f'value = "{PW}"', PW),  # password-shaped literal with a neutral name
    ('value = "hunter4242"', "hunter4242"),  # weak lowercase+digits password
])
def test_credential_shapes_become_ml_candidates(line, expected):
    assert expected in _ml_tokens(line)


@pytest.mark.parametrize("line", [
    'label = "Content-Type"',        # no digit
    'version = "1.2.3"',             # no letter mix / password symbol
    'path = "src/app/main.py"',      # path
    'name = "userProfileHandler"',   # identifier
])
def test_plain_strings_are_not_candidates(line):
    assert not _ml_tokens(line)


def test_non_ml_scan_reports_no_new_candidates():
    line = f'value = "{PW}"'
    assert not detect_text(line)


def test_long_dotted_run_without_scheme_is_fast():
    import time

    line = 'x = "' + ".".join(["ab"] * 60_000) + '"'
    start = time.perf_counter()
    _collect_text_findings(line)
    assert time.perf_counter() - start < 2.0
