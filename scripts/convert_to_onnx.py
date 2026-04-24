#!/usr/bin/env python3
"""
Offline ONNX model conversion script (v0.2.0).

Converts the trained XGBoost (Stage A, 23 features) and LightGBM (Stage B,
63 features) models to ONNX format for lightweight runtime inference.

Outputs:
    Harpocrates/ml/models/stageA.onnx
    Harpocrates/ml/models/stageB.onnx
    Harpocrates/ml/models/onnx_model_hashes.json

Usage:
    python scripts/convert_to_onnx.py
    python scripts/convert_to_onnx.py --model-dir /path/to/models

Requirements (pip install harpocrates[ml]):
    xgboost>=2.0.0, lightgbm>=4.0.0, onnxmltools>=1.12.0, skl2onnx>=0.5.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_DIR = ROOT / "Harpocrates" / "ml" / "models"

# Feature counts must match training config (two_stage_config.json v2.4.0)
STAGE_A_N_FEATURES = 23   # token-only features
STAGE_B_N_FEATURES = 63   # all features


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_xgboost(model_path: Path, output_path: Path, n_features: int) -> None:
    """Convert XGBoost JSON model → ONNX via XGBClassifier + onnxmltools."""
    print(f"[Stage A] Converting XGBoost: {model_path.name} → {output_path.name}")

    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install 'harpocrates[ml]' onnxmltools skl2onnx")
        sys.exit(1)

    # XGBClassifier.load_model() supports native .json format (XGBoost ≥ 1.6)
    clf = XGBClassifier()
    clf.load_model(str(model_path))

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_types)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    kb = output_path.stat().st_size // 1024
    print(f"  Saved {output_path} ({kb} KB)")


def convert_lightgbm(model_path: Path, output_path: Path, n_features: int) -> None:
    """Convert LightGBM text model → ONNX via onnxmltools."""
    print(f"[Stage B] Converting LightGBM: {model_path.name} → {output_path.name}")

    try:
        import lightgbm as lgb
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install 'harpocrates[ml]' onnxmltools skl2onnx")
        sys.exit(1)

    booster = lgb.Booster(model_file=str(model_path))
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(booster, initial_types=initial_types)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    kb = output_path.stat().st_size // 1024
    print(f"  Saved {output_path} ({kb} KB)")


def smoke_test(stage_a_path: Path, stage_b_path: Path) -> None:
    """Quick single-row inference smoke test to verify converted models."""
    print("\nRunning smoke test...")
    try:
        import numpy as np
        import onnxruntime as ort

        for path, n_feat in [(stage_a_path, STAGE_A_N_FEATURES), (stage_b_path, STAGE_B_N_FEATURES)]:
            session = ort.InferenceSession(str(path))
            dummy = np.zeros((1, n_feat), dtype=np.float32)
            result = session.run(None, {session.get_inputs()[0].name: dummy})
            print(f"  {path.name}: OK (output shapes: {[r.shape if hasattr(r, 'shape') else type(r).__name__ for r in result]})")
    except Exception as e:
        print(f"  [WARN] Smoke test failed: {e}")


def write_hash_manifest(hashes: dict, manifest_path: Path) -> None:
    """Write SHA-256 hash manifest for supply-chain verification."""
    with open(manifest_path, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"\nHash manifest: {manifest_path}")
    for name, digest in hashes.items():
        print(f"  {name}: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ML models to ONNX (v0.2.0)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Directory containing source and output models")
    parser.add_argument("--stage-a-input", type=Path, default=None,
                        help="Path to stageA_xgboost.json (default: <model-dir>/stageA_xgboost.json)")
    parser.add_argument("--stage-b-input", type=Path, default=None,
                        help="Path to stageB_lightgbm.txt (default: <model-dir>/stageB_lightgbm.txt)")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Skip inference smoke test after conversion")
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    stage_a_in = args.stage_a_input or model_dir / "stageA_xgboost.json"
    stage_b_in = args.stage_b_input or model_dir / "stageB_lightgbm.txt"
    stage_a_out = model_dir / "stageA.onnx"
    stage_b_out = model_dir / "stageB.onnx"
    manifest_path = model_dir / "onnx_model_hashes.json"

    # Pre-flight: ensure source models exist
    missing = [p for p in (stage_a_in, stage_b_in) if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: Model file not found: {p}")
        sys.exit(1)

    convert_xgboost(stage_a_in, stage_a_out, STAGE_A_N_FEATURES)
    convert_lightgbm(stage_b_in, stage_b_out, STAGE_B_N_FEATURES)

    if not args.no_smoke_test:
        smoke_test(stage_a_out, stage_b_out)

    # Compute and persist SHA-256 hashes (supply-chain verification manifest)
    hashes = {
        stage_a_out.name: _sha256(stage_a_out),
        stage_b_out.name: _sha256(stage_b_out),
    }
    write_hash_manifest(hashes, manifest_path)

    print("\nConversion complete.")
    print("Add stageA.onnx, stageB.onnx, onnx_model_hashes.json to git and re-build the wheel.")


if __name__ == "__main__":
    main()
