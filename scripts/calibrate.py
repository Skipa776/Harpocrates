"""Fit the FR-CORE-03 calibration map on validation candidates and report it on held-out sets.

    python scripts/calibrate.py            # measure only
    python scripts/calibrate.py --write    # also write the map into the shipped model_config.json

The map is isotonic regression from the shipped ONNX score to P(secret), fit on the validation split
only (DATA-09). test_v5 and benchmark_llm_v5 are scored with it, never fit on. Routing thresholds stay
on the raw score, so this changes the reported confidence, never which findings are kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]

import r2_common as C  # noqa: E402
import train_v11 as T  # noqa: E402

MODELS = ROOT / "cli/Harpocrates/ml/models"
BINS = 10
MIN_PLOTTED = 20  # a 2-candidate bin is noise, not a calibration signal


def onnx_scores(X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    from Harpocrates.ml.onnx_verifier import _run_session

    session = ort.InferenceSession(str(MODELS / "model.onnx"))
    return np.array([p for i in range(0, len(X), 4096) for p in _run_session(session, X[i:i + 4096].tolist())])


def reliability(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict]]:
    """Expected calibration error over 10 equal-width bins, and the bins themselves."""
    idx = np.minimum((p * BINS).astype(int), BINS - 1)
    out = [{"lo": b / BINS, "n": int((idx == b).sum()), "mean_p": float(p[idx == b].mean()),
            "frac_pos": float(y[idx == b].mean())} for b in range(BINS) if (idx == b).any()]
    return sum(b["n"] * abs(b["frac_pos"] - b["mean_p"]) for b in out) / len(p), out


def fit(y: np.ndarray, p: np.ndarray) -> tuple[list[float], list[float]]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    xs, ys = np.round(iso.X_thresholds_, 4), np.round(iso.y_thresholds_, 4)
    xs, keep = np.unique(xs, return_index=True)  # rounding can merge breakpoints; keep the first
    xs, ys = list(xs), list(np.maximum.accumulate(ys[keep]))
    if xs[0] > 0.0:
        xs, ys = [0.0, *xs], [ys[0], *ys]
    if xs[-1] < 1.0:
        xs, ys = [*xs, 1.0], [*ys, ys[-1]]
    return [float(v) for v in xs], [float(v) for v in ys]


def plot(curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4.5), sharey=True)
    for ax, (name, (raw, cal)) in zip(np.atleast_1d(axes), curves.items()):
        ax.plot([0, 1], [0, 1], ":", color="grey", label="perfect")
        for label, bins in (("raw score", raw), ("calibrated", cal)):
            bins = [b for b in bins if b["n"] >= MIN_PLOTTED]
            ax.plot([b["mean_p"] for b in bins], [b["frac_pos"] for b in bins], "o-", label=label)
        ax.set_title(f"{name} (bins with {MIN_PLOTTED}+ candidates)")
        ax.set_xlabel("predicted P(secret)")
        ax.legend(loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("fraction that are secrets")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the map into model_config.json")
    args = parser.parse_args()

    _, val = C.split()
    sets = {"val (fit)": val, "test_v5": T.load(C.E / "test_v5.jsonl"),
            "benchmark_llm_v5": T.load(C.E / "benchmark_llm_v5.jsonl")}
    scored = {}
    for name, recs in sets.items():
        d = C.rows(recs)
        scored[name] = (d["y"], onnx_scores(d["X"]))
    xs, ys = fit(*scored["val (fit)"])

    report, curves = {"map_points": len(xs)}, {}
    print(f"map: {len(xs)} points\n\n| Set | Candidates | ECE raw | ECE calibrated |\n| --- | --- | --- | --- |")
    for name, (y, p) in scored.items():
        e_raw, b_raw = reliability(y, p)
        e_cal, b_cal = reliability(y, np.interp(p, xs, ys))
        report[name] = {"candidates": len(y), "ece_raw": e_raw, "ece_calibrated": e_cal, "bins": b_cal}
        curves[name] = (b_raw, b_cal)
        print(f"| {name} | {len(y):,} | {e_raw:.4f} | {e_cal:.4f} |")

    (ROOT / "data/models").mkdir(exist_ok=True)
    (ROOT / "data/models/calibration.json").write_text(json.dumps(report, indent=1))
    plot({k: v for k, v in curves.items() if k != "val (fit)"}, ROOT / "docs/assets/calibration.png")
    if args.write:
        cfg_path = MODELS / "model_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["calibration"] = {"method": "isotonic", "fit_on": "validation split candidates",
                              "x": xs, "y": ys}
        cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
        from convert_to_onnx import _sha256, write_hash_manifest

        write_hash_manifest({p.name: _sha256(p) for p in (MODELS / "model.onnx", cfg_path)},
                            MODELS / "onnx_model_hashes.json")


if __name__ == "__main__":
    main()
