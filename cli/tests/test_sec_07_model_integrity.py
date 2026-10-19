"""SEC-07: every model artifact the package loads is hash-checked against the manifest.

A missing manifest entry or a mismatched hash is refused, never loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List

import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

from Harpocrates.ml import explain as explain_mod  # noqa: E402
from Harpocrates.ml.explain import _load_booster, _reset_booster_cache  # noqa: E402
from Harpocrates.ml.onnx_verifier import ONNX_HASHES_PATH  # noqa: E402

SHIPPED_XGBOOST = explain_mod._DEFAULT_XGBOOST_PATH
XGBOOST_KEY = "xgboost_model.json"


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Keep the process-wide Booster cache from leaking between tests."""
    _reset_booster_cache()
    yield
    _reset_booster_cache()


@pytest.fixture
def load_calls(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Record every Booster.load_model call without parsing anything."""
    calls: List[Any] = []

    def _record(self: Any, fname: Any) -> None:
        calls.append(fname)

    monkeypatch.setattr(xgb.Booster, "load_model", _record)
    return calls


def _sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict[str, str] | None
) -> Path:
    """Copy the shipped model into tmp_path and point the loader at tmp_path."""
    model = tmp_path / XGBOOST_KEY
    model.write_bytes(SHIPPED_XGBOOST.read_bytes())
    hashes = tmp_path / "onnx_model_hashes.json"
    if manifest is not None:
        hashes.write_text(json.dumps(manifest))
    monkeypatch.setattr(explain_mod, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(explain_mod, "_DEFAULT_HASHES_PATH", hashes)
    return model


def _real_hash() -> str:
    return hashlib.sha256(SHIPPED_XGBOOST.read_bytes()).hexdigest()


def test_sec_07_shipped_xgboost_model_matches_manifest() -> None:
    manifest = json.loads(ONNX_HASHES_PATH.read_text())
    assert manifest.get(XGBOOST_KEY) == _real_hash()
    booster = _load_booster(None)
    assert booster.num_features() > 0


@pytest.mark.parametrize("tamper", ["append", "flip"])
def test_sec_07_tampered_xgboost_model_refused(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_calls: List[Any],
) -> None:
    model = _sandbox(tmp_path, monkeypatch, {XGBOOST_KEY: _real_hash()})
    data = bytearray(model.read_bytes())
    if tamper == "append":
        data += b" "
    else:
        data[len(data) // 2] ^= 0x01
    model.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_booster(model)

    assert load_calls == [], "tampered model reached Booster.load_model"
    assert explain_mod._BOOSTER_CACHE == {}


def test_sec_07_xgboost_model_missing_from_manifest_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_calls: List[Any]
) -> None:
    model = _sandbox(tmp_path, monkeypatch, {"model.onnx": "0" * 64})

    with pytest.raises(ValueError, match="does not contain an entry"):
        _load_booster(model)

    assert load_calls == []
    assert explain_mod._BOOSTER_CACHE == {}


def test_sec_07_xgboost_manifest_missing_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_calls: List[Any]
) -> None:
    model = _sandbox(tmp_path, monkeypatch, None)

    with pytest.raises(FileNotFoundError, match="Refusing to load"):
        _load_booster(model)

    assert load_calls == []


def test_sec_07_xgboost_loaded_from_verified_bytes_not_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_calls: List[Any]
) -> None:
    """The buffer that was hashed is the buffer that is loaded (no check-then-reopen gap)."""
    model = _sandbox(tmp_path, monkeypatch, {XGBOOST_KEY: _real_hash()})

    _load_booster(model)

    assert len(load_calls) == 1
    assert isinstance(load_calls[0], bytearray)
    assert bytes(load_calls[0]) == model.read_bytes()
