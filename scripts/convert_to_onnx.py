#!/usr/bin/env python3
"""
Offline ONNX model conversion script (v2.1.0).

Converts the trained XGBoost (65 features) model to ONNX format
for lightweight runtime inference.

Outputs:
    Harpocrates/ml/models/model.onnx
    Harpocrates/ml/models/onnx_model_hashes.json

Usage:
    python scripts/convert_to_onnx.py
    python scripts/convert_to_onnx.py --model-dir /path/to/models

Requirements (pip install harpocrates[ml]):
    xgboost>=2.0.0, onnxmltools>=1.12.0, skl2onnx>=0.5.0
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

N_FEATURES = 65


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_xgboost(model_path: Path, output_path: Path, n_features: int) -> None:
    """Convert XGBoost JSON model to ONNX."""
    print(f"Converting XGBoost: {model_path.name} -> {output_path.name}")

    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install 'harpocrates[ml]' onnxmltools skl2onnx")
        sys.exit(1)

    clf = XGBClassifier()
    clf.load_model(str(model_path))

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_types)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    kb = output_path.stat().st_size // 1024
    print(f"  Saved {output_path} ({kb} KB)")


def smoke_test(model_path: Path) -> None:
    """Quick single-row inference smoke test. Raises on failure."""
    print("\nRunning smoke test...")
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    dummy = np.zeros((1, N_FEATURES), dtype=np.float32)
    result = session.run(None, {session.get_inputs()[0].name: dummy})
    shapes = [
        r.shape if hasattr(r, "shape") else type(r).__name__ for r in result
    ]
    print(f"  {model_path.name}: OK (output shapes: {shapes})")


def write_hash_manifest(hashes: dict, manifest_path: Path) -> None:
    """Write SHA-256 hash manifest for supply-chain verification."""
    with open(manifest_path, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"\nHash manifest: {manifest_path}")
    for name, digest in hashes.items():
        print(f"  {name}: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XGBoost model to ONNX (v2.1.0)")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing source and output models",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to stageA_xgboost.json (default: <model-dir>/stageA_xgboost.json)",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip inference smoke test after conversion",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    model_in = args.input or model_dir / "stageA_xgboost.json"
    model_out = model_dir / "model.onnx"
    manifest_path = model_dir / "onnx_model_hashes.json"

    if not model_in.exists():
        print(f"ERROR: Model file not found: {model_in}")
        sys.exit(1)

    convert_xgboost(model_in, model_out, N_FEATURES)

    if not args.no_smoke_test:
        try:
            smoke_test(model_out)
        except Exception as e:
            print(f"\nERROR: Smoke test failed: {e}")
            print("ONNX model is broken. Not writing hash manifest.")
            sys.exit(1)

    hashes = {model_out.name: _sha256(model_out)}
    write_hash_manifest(hashes, manifest_path)

    print("\nConversion complete.")
    print("Add model.onnx and onnx_model_hashes.json to git and re-build the wheel.")


if __name__ == "__main__":
    main()
