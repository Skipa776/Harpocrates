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


# Round 2 (bench/model_improvement.ipynb section 14): shapes found uncovered in LLM-written code.
SOUP = "Qm" + "]a%)A-x;7" + "Kp{9!z"
TOK59 = "MTk4NjIyNDgzNDcxOTI1MjQ4" + "." + "Cl2FMQ" + "." + "ZGiU5uKqX0vWf7Q2pRt8sLm1nBcD"


@pytest.mark.parametrize("line, expected", [
    ('-H "Authorization: Bearer ab9Kd_Wq2x"', "ab9Kd_Wq2x"),                       # short bearer token
    ('--header "X-Deploy-Key: Zk2+mQ9vL4xP7wR1tY8n" \\', "Zk2+mQ9vL4xP7wR1tY8n"),  # custom header
    (f'hook = "https://discord.com/api/webhooks/{TOK59}"', TOK59),                  # token as a URL path segment
    ('u = "https://api.example.com/v2/stock?session=aZ8kQ2mW9xR4tY7uP1vB3nC6&w=chi3"',
     "aZ8kQ2mW9xR4tY7uP1vB3nC6"),                                                  # any query parameter name
    ('dsn = "https://4f9Kx2Qm8vL1pR7tZ3wY@o1.ingest.sentry.io/450"', "4f9Kx2Qm8vL1pR7tZ3wY"),  # key-only userinfo
    (f"bootstrap_value = {SOUP}", SOUP),                                            # symbol-heavy random value
    (f'WEEKLY = "{SOUP}"', SOUP),
    ("x https://h.io/AbC9xYz1QwErTy7uIoPz3K8mN==?a=1", "AbC9xYz1QwErTy7uIoPz3K8mN=="),
    ("x https://h.io/?key=Zx9Qw7Er5Ty3Ui1Op8As&b=1", "Zx9Qw7Er5Ty3Ui1Op8As"),
    ("see (https://h.io/p?sess=Zx9Qw7Er5Ty3Ui1Op8As).", "Zx9Qw7Er5Ty3Ui1Op8As"),
    ("see HTTPS://h.io/v1/Zx9Qw7Er5Ty3Ui1Op8AsLk", "Zx9Qw7Er5Ty3Ui1Op8AsLk"),
])
def test_round2_shapes_become_ml_candidates(line, expected):
    """Also covers review fixes: padded path segments, a query right after the scheme, trailing punctuation."""
    assert expected in {f.token for f in _collect_text_findings(line) if f.evidence != EvidenceType.REGEX}


@pytest.mark.parametrize("line", [
    'url = "https://github.com/Skipa776/Harpocrates/blob/main/README.md"',  # word path segments
    'pattern = r"^[a-z]+$"',                                                # short regex
    'fmt = "%Y-%m-%d %H:%M:%S"',                                            # has spaces
])
def test_round2_plain_strings_are_not_candidates(line):
    assert not {f.token for f in _collect_text_findings(line) if f.evidence != EvidenceType.REGEX}
