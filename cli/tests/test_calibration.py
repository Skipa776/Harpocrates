"""FR-CORE-03: ML findings carry a calibrated confidence, P(secret), from a map fit on validation."""
from __future__ import annotations

import json

import numpy as np
import pytest

from Harpocrates.ml.onnx_verifier import (
    MODEL_CONFIG_PATH,
    OnnxModelSchemaError,
    _calibration_map,
    _validate_model_config,
)


def test_fr_core_03_shipped_map_is_monotone_probability() -> None:
    xs, ys = _calibration_map(json.loads(MODEL_CONFIG_PATH.read_text()))
    assert xs[0] == 0.0 and xs[-1] == 1.0
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) >= 0)
    assert 0.0 <= ys[0] and ys[-1] <= 1.0


def test_fr_core_03_verifier_applies_map() -> None:
    from Harpocrates.core.detector import _collect_text_findings, _prepare_ml_context_from_lines
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    lines = ['DB_PASSWORD = "Kq7#vR2pLm9xTz4w"', 'api_key = get_key("name")']
    pairs = [(f, _prepare_ml_context_from_lines(f, lines)) for f in _collect_text_findings("\n".join(lines))]
    verifier = OnnxVerifier(layer="gate")
    results = verifier.verify_batch(pairs)
    assert results
    xs, ys = _calibration_map(json.loads(MODEL_CONFIG_PATH.read_text()))
    for r in results:
        raw = r.ml_confidence if r.is_secret else 1.0 - r.ml_confidence
        assert r.calibrated == pytest.approx(float(np.interp(raw, xs, ys)))


def test_fr_core_03_finding_confidence_is_calibrated() -> None:
    from Harpocrates.core.detector import detect_text_with_ml
    from Harpocrates.ml.verifier import VerificationResult

    class Stub:
        def verify_batch(self, batch):
            return [VerificationResult(True, 0.9, 0.5, 0.8, calibrated=0.42) for _ in batch]

    findings = detect_text_with_ml('DB_PASSWORD = "Kq7#vR2pLm9xTz4w"\n', Stub(), ml_threshold=0.0)
    ml = [f for f in findings if f.evidence.value == "hybrid"]
    assert ml and all(f.confidence == 0.42 for f in ml)


def test_fr_core_03_bad_map_refused() -> None:
    base = {"threshold_low": 0.5, "threshold_high": 0.95}
    for bad in ({"x": [0.0, 1.0], "y": [0.9, 0.1]},  # decreasing
                {"x": [0.0, 0.5], "y": [0.0, 1.0]},  # does not cover [0, 1]
                {"x": [0.0, 1.0], "y": [0.0, 1.5]},  # not a probability
                {"x": [0.0, 1.0], "y": [0.0]}):  # length mismatch
        with pytest.raises(OnnxModelSchemaError, match="calibration"):
            _calibration_map({**base, "calibration": bad})
    assert _calibration_map(base) is None  # no map: report the raw score
    _validate_model_config(base)


def test_fr_core_03_map_and_platt_refused_together() -> None:
    cfg = {"threshold_low": 0.5, "threshold_high": 0.95, "calibration": [0.0, 1.0]}
    with pytest.raises(OnnxModelSchemaError, match="calibration"):
        _calibration_map(cfg)
    cfg["calibration"] = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    with pytest.raises(OnnxModelSchemaError, match="Platt"):
        _calibration_map({**cfg, "platt_a": 1.0})
