"""Contracts for the shipped ONNX verifier artifact."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from Harpocrates.core.detector import detect_text_with_ml
from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FEATURE_NAMES
from Harpocrates.ml.onnx_verifier import (
    MODEL_CONFIG_PATH,
    OnnxModelSchemaError,
    OnnxVerifier,
    _run_session,
    _validate_model_config,
    _validate_model_schema,
)


def test_shipped_onnx_schema_matches_feature_contract() -> None:
    """The runtime, model config, and feature extractor must agree on width."""
    verifier = OnnxVerifier(lazy_load=False)
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))

    assert verifier.input_feature_count == len(FEATURE_NAMES)
    assert config["feature_count"] == len(FEATURE_NAMES)


def test_shipped_onnx_verifier_executes_real_feature_vector() -> None:
    """A real 64-wide feature vector must complete ONNX inference."""
    token = "aB3dEfGhIjKlMnOpQrStUvWxYz012345"
    finding = Finding(
        type="ENTROPY_CANDIDATE",
        snippet=f'api_secret = "{token}"',
        evidence=EvidenceType.ENTROPY,
        token=token,
        confidence=0.7,
    )
    context = CodeContext(
        line_content=f'api_secret = "{token}"',
        file_path="config.py",
    )

    result = OnnxVerifier(lazy_load=False).verify(finding, context)

    assert 0.0 <= result.ml_confidence <= 1.0
    assert result.explanation is not None
    assert "ONNX/" in result.explanation


def test_real_entropy_candidate_becomes_hybrid() -> None:
    """The end-to-end ML path must execute ONNX and mark accepted evidence."""
    text = 'api_secret = "aB3dEfGhIjKlMnOpQrStUvWxYz012345"\n'

    findings = detect_text_with_ml(
        text,
        verifier=OnnxVerifier(lazy_load=False),
        ml_threshold=0.0,
    )

    assert findings
    assert findings[0].evidence == EvidenceType.HYBRID


def test_onnx_schema_mismatch_fails_before_inference() -> None:
    """A stale artifact must be rejected with expected and actual widths."""
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="features", shape=[None, 65])]
    )

    with pytest.raises(OnnxModelSchemaError, match=r"expects 65.*emits 64"):
        _validate_model_schema(session, configured_feature_count=64)


def test_onnx_verifier_rejects_tampered_model_before_inference(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"tampered-model")
    hashes_path = tmp_path / "onnx_model_hashes.json"
    hashes_path.write_text(
        json.dumps({"model.onnx": hashlib.sha256(b"original-model").hexdigest()}),
        encoding="utf-8",
    )
    config_path = tmp_path / "model_config.json"
    config_path.write_text(json.dumps({"feature_count": 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        OnnxVerifier(
            model_path=model_path,
            hashes_path=hashes_path,
            config_path=config_path,
            lazy_load=False,
        )


def test_onnx_verifier_requires_hash_manifest(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    config_path = tmp_path / "model_config.json"
    config_path.write_text(json.dumps({"feature_count": 64}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="hash manifest"):
        OnnxVerifier(
            model_path=model_path,
            hashes_path=tmp_path / "missing-hashes.json",
            config_path=config_path,
            lazy_load=False,
        )


def test_onnx_verifier_requires_policy_config(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    hashes_path = tmp_path / "onnx_model_hashes.json"
    hashes_path.write_text(
        json.dumps({"model.onnx": hashlib.sha256(b"model").hexdigest()}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="model config"):
        OnnxVerifier(
            model_path=model_path,
            hashes_path=hashes_path,
            config_path=tmp_path / "missing-config.json",
            lazy_load=False,
        )


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_onnx_session_rejects_invalid_probabilities(probability) -> None:
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="features")],
        run=lambda *args, **kwargs: [
            [0],
            [[1.0 - probability, probability]],
        ],
    )

    with pytest.raises(ValueError, match="probability"):
        _run_session(session, [[0.0] * len(FEATURE_NAMES)])


@pytest.mark.parametrize(
    "config",
    [
        {"threshold_low": 2.0, "threshold_high": 0.85},
        {"threshold_low": 0.9, "threshold_high": 0.1},
        {"threshold_low": True, "threshold_high": 0.85},
        {"threshold_low": 0.15, "threshold_high": float("nan")},
        {"threshold_low": 0.15, "threshold_high": 0.85, "platt_a": "bad"},
        {"feature_count": True},
    ],
)
def test_onnx_model_config_rejects_unsafe_routing_values(config) -> None:
    with pytest.raises(OnnxModelSchemaError):
        _validate_model_config(config)
