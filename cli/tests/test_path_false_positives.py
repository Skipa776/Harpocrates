"""Filesystem paths are not secrets: long absolute/relative paths in scripts and commands must not be
redacted by the read gate, while secrets that sit inside paths or URLs still are.

Every random-looking value here (UUIDs, hashes, tokens, keys) is generated at runtime from a seeded RNG,
so no literal in this file is a credential and results are reproducible.
"""
from __future__ import annotations

import random
import string
import uuid
from typing import List, Tuple

import pytest

from Harpocrates.core.detector import _apply_ml_verification, _collect_text_findings
from Harpocrates.core.result import EvidenceType
from Harpocrates.detectors.path_tokens import benign_path_token
from Harpocrates.ml.verifier import VerificationResult, Verifier

_RNG = random.Random(20261012)
_LOWER_ALNUM = string.ascii_lowercase + string.digits
_B62 = string.ascii_letters + string.digits
_HEX = "0123456789abcdef"


def _pick(alphabet: str, k: int) -> str:
    return "".join(_RNG.choices(alphabet, k=k))


_UUID = str(uuid.UUID(int=_RNG.getrandbits(128)))
_HEX56 = _pick(_HEX, 56)
_VAR_FOLDER = _pick(_LOWER_ALNUM, 24) + "0000gn"  # macOS /var/folders/xx/<this>/T
_PY_TMP = "tmp" + _pick(_LOWER_ALNUM + "_", 8)  # tempfile.mkdtemp() name
_PNPM_HASH = _pick(_LOWER_ALNUM, 26)  # pnpm peer-dependency suffix
_XCODE_HASH = _pick(string.ascii_lowercase, 28)  # Xcode DerivedData suffix
_TOKEN40 = _pick(_B62, 40)  # generic machine token, mixed case + digits
_SIG = _pick(_HEX, 64)  # SigV4 signature
_AKIA = "AKIA" + _pick(string.ascii_uppercase + "234567", 16)
_B64_SLASHED = _pick(_B62, 12) + "/" + _pick(_B62, 13) + "/" + _pick(_B62, 13)  # 40-char base64 with slashes

PATH_LINES = {
    "uuid_temp_dir": f"cd /private/tmp/claude-501/-Users-dev-myproject/{_UUID}/scratchpad/live\n",
    "uuid_temp_dir_quoted": f'OUT="/private/tmp/claude-501/-Users-dev-myproject/{_UUID}/scratchpad/out.txt"\n',
    "pip_cache": f"ls ~/.cache/pip/http/a/1/b/2/{_HEX56}\n",
    "xcode_derived_data": (
        f'path = "/Users/dev/Library/Caches/com.apple.dt.Xcode/DerivedData/MyApp-{_XCODE_HASH}/Build"\n'
    ),
}

# Recall first: a path segment mixing letters and digits at secret-level entropy keeps the path a
# candidate even when it is a generated directory name; the model decides these, not the path rule.
RANDOM_SEGMENT_PATH_LINES = {
    "macos_var_folders": f"TMP=/var/folders/xq/{_VAR_FOLDER}/T/{_PY_TMP}/build.log\n",
    "pnpm_store": (
        "import x from \"node_modules/.pnpm/react-dom@18.3.1_react@18.3.1_"
        f"{_PNPM_HASH}/node_modules/react-dom/index.js\"\n"
    ),
}

SECRET_LINES = {
    "url_path_token": f"WEBHOOK = 'https://hooks.example.com/hook/{_TOKEN40}'\n",
    "s3_presigned_signature": f"url = 's3://bucket/key?X-Amz-Signature={_SIG}'\n",
    "aws_key_as_path_segment": f"cp data.csv /mnt/{_AKIA}/config/data.csv\n",
    "token_as_plain_path_segment": f"curl -s localhost/api/v1/key/{_TOKEN40}/status\n",
    "base64_secret_with_slashes": f"aws_secret_access_key = {_B64_SLASHED}\n",
}


class _AcceptAll(Verifier):
    """Says every candidate is a secret, so whatever is dropped was dropped before the model."""

    def verify(self, finding, context) -> VerificationResult:
        return VerificationResult(True, 1.0, 1.0, 1.0)

    def verify_batch(self, findings_with_context) -> List[VerificationResult]:
        return [self.verify(f, c) for f, c in findings_with_context]

    @property
    def is_loaded(self) -> bool:
        return True


def _verified_entropy_tokens(text: str) -> List[str]:
    candidates = [f for f in _collect_text_findings(text) if f.evidence != EvidenceType.REGEX]
    verified = _apply_ml_verification(candidates, text, _AcceptAll(), ml_threshold=0.0)
    return [f.token for f in verified if f.type == "ENTROPY_CANDIDATE"]


def _entropy_tokens(text: str) -> List[str]:
    return [f.token for f in _collect_text_findings(text) if f.type == "ENTROPY_CANDIDATE"]


# --- the rule itself -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PATH_LINES))
def test_path_entropy_tokens_are_benign(name: str) -> None:
    line = PATH_LINES[name]
    tokens = _entropy_tokens(line)
    assert all(benign_path_token(t, line) for t in tokens), name


@pytest.mark.parametrize("name", sorted(RANDOM_SEGMENT_PATH_LINES))
def test_letters_and_digits_segment_keeps_the_path_a_candidate(name: str) -> None:
    line = RANDOM_SEGMENT_PATH_LINES[name]
    tokens = [t for t in _entropy_tokens(line) if "/" in t]
    assert tokens and not any(benign_path_token(t, line) for t in tokens), name


def test_base64_with_slashes_is_not_a_path() -> None:
    line = f"aws_secret_access_key = {_B64_SLASHED}\n"
    assert not benign_path_token(_B64_SLASHED, line)


def test_padded_base64_is_not_a_path() -> None:
    token = "config/" + _pick(_B62, 20) + "/data=="
    assert not benign_path_token(token, f"x = {token}\n")


def test_path_with_high_entropy_segment_stays_a_candidate() -> None:
    token = f"api/v1/key/{_TOKEN40}/status"
    assert not benign_path_token(token, f"curl -s localhost/{token}\n")


_B32_UPPER = _pick(string.ascii_uppercase + "234567", 32)  # base32 token: uppercase + digits, no lowercase


def test_base32_uppercase_segment_keeps_the_candidate() -> None:
    token = f"home/user/data/{_B32_UPPER}"
    assert not benign_path_token(token, f"cat /{token}\n")


def test_base32_uppercase_segment_still_reaches_the_model() -> None:
    assert f"/home/user/data/{_B32_UPPER}" in _verified_entropy_tokens(f"cat /home/user/data/{_B32_UPPER}\n")


@pytest.mark.parametrize("key", ["secret", "token", "api_key", "password", "credential", "auth"])
def test_secret_named_key_before_path_keeps_the_candidate(key: str) -> None:
    token = f"srv/keys/{_UUID}/current"
    assert not benign_path_token(token, f"{key}: /{token}\n")
    assert not benign_path_token(token, f"{key.upper()}=/{token}\n")


def test_url_lines_are_left_to_the_url_rules() -> None:
    token = f"//files.example.com/private/tmp/{_UUID}/scratchpad"
    assert not benign_path_token(token, f"GET https:{token}\n")


def test_query_after_path_keeps_the_candidate() -> None:
    token = f"private/tmp/claude-501/{_UUID}/scratchpad"
    assert not benign_path_token(token, f"/{token}?token=abc\n")


def test_single_segment_is_not_a_path() -> None:
    token = _pick(_LOWER_ALNUM, 30)
    assert not benign_path_token(token, f"x = {token}\n")


def test_one_word_segment_is_not_enough_evidence() -> None:
    token = "data/" + _pick(_B62, 14) + "/" + _pick(_B62, 14)
    assert not benign_path_token(token, f"value = {token}\n")


# --- where it applies: entropy candidates, before the model ----------------------------------------


@pytest.mark.parametrize("name", sorted(PATH_LINES))
def test_path_candidates_never_reach_the_model(name: str) -> None:
    assert _verified_entropy_tokens(PATH_LINES[name]) == [], name


def test_base64_secret_with_slashes_still_reaches_the_model() -> None:
    assert _B64_SLASHED in _verified_entropy_tokens(SECRET_LINES["base64_secret_with_slashes"])


def test_regex_tier_is_untouched_by_the_path_rule() -> None:
    findings = _collect_text_findings(SECRET_LINES["aws_key_as_path_segment"])
    assert any(f.evidence == EvidenceType.REGEX and f.token == _AKIA for f in findings)


# --- end to end through the read gate ---------------------------------------------------------------


@pytest.fixture(scope="module")
def redactor():
    pytest.importorskip("onnxruntime")
    from Harpocrates.read import Redactor

    return Redactor()


@pytest.mark.parametrize("name", sorted(PATH_LINES))
def test_read_gate_leaves_paths_alone(redactor, name: str) -> None:
    line = PATH_LINES[name]
    assert redactor.redact(line) == line, name


@pytest.mark.parametrize("name", sorted(SECRET_LINES))
def test_read_gate_still_redacts_secrets_in_paths_and_urls(redactor, name: str) -> None:
    out = redactor.redact(SECRET_LINES[name])
    assert "<<HARPO:" in out, name
    secret: Tuple[str, ...] = (_TOKEN40, _SIG, _AKIA, _B64_SLASHED)
    assert not any(s in out for s in secret), name
