#!/usr/bin/env python3
"""
Training experiments v3 - Aggressive precision optimization.

Analysis from v2:
- FPs have: lower token_length, higher is_base64_like, higher is_known_hash_length
- FPs are likely: short base64 strings, hash-length tokens

Strategy:
1. Generate more hard negatives (60% of negatives)
2. Use stricter class weights
3. Add post-hoc calibration
4. Try different Stage A/B splits
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load training data from pickle file."""
    from Harpocrates.ml.features import extract_features_from_record

    with open(data_path, "rb") as f:
        records = pickle.load(f)

    features = []
    labels = []
    valid_records = []

    for record in records:
        try:
            fv = extract_features_from_record(record)
            features.append(fv.to_array())
            labels.append(record["label"])
            valid_records.append(record)
        except Exception:
            continue

    return np.array(features), np.array(labels), valid_records


def get_token_feature_indices() -> List[int]:
    """Token-level features (first 23)."""
    return list(range(23))


def generate_more_hard_negatives(count: int = 10000) -> List[Dict]:
    """Generate dataset with higher hard negative ratio."""
    from Harpocrates.training.generators.generate_data import generate_training_data

    # Generate with 50% balance
    records = generate_training_data(count=count, balance=0.5, seed=123)
    return records


def train_precision_focused(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Train with precision-focused hyperparameters."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    token_indices = get_token_feature_indices()
    X_train_tokens = X_train[:, token_indices]
    X_val_tokens = X_val[:, token_indices]

    results = []

    # Define configurations to test
    configs = [
        # (stage_a_low, stage_a_high, stage_a_scale, stage_b_scale)
        {"name": "baseline", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.8, "b_depth": 8},
        {"name": "strict_b", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.5, "b_depth": 10},
        {"name": "very_strict_b", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.3, "b_depth": 12},
        {"name": "narrow_a", "a_low": 0.25, "a_high": 0.75, "a_scale": 1.5, "b_scale": 0.5, "b_depth": 10},
        {"name": "wide_a_strict_b", "a_low": 0.10, "a_high": 0.90, "a_scale": 1.8, "b_scale": 0.4, "b_depth": 12},
        {"name": "balanced", "a_low": 0.20, "a_high": 0.80, "a_scale": 1.3, "b_scale": 0.6, "b_depth": 10},
        {"name": "ultra_strict_b", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.2, "b_depth": 14},
        {"name": "precision_tuned", "a_low": 0.12, "a_high": 0.88, "a_scale": 1.6, "b_scale": 0.35, "b_depth": 12},
    ]

    for cfg in configs:
        print(f"\nTesting: {cfg['name']}")

        n_pos = sum(y_train)
        n_neg = len(y_train) - n_pos

        # Train Stage A
        stage_a = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=5,
            learning_rate=0.1,
            n_estimators=120,
            min_child_weight=3,
            scale_pos_weight=(n_neg / n_pos) * cfg["a_scale"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        stage_a.fit(X_train_tokens, y_train, verbose=False)

        stage_a_train_proba = stage_a.predict_proba(X_train_tokens)[:, 1]
        stage_a_val_proba = stage_a.predict_proba(X_val_tokens)[:, 1]

        # Filter ambiguous for Stage B training
        amb_mask = (stage_a_train_proba >= cfg["a_low"]) & (stage_a_train_proba <= cfg["a_high"])
        if sum(amb_mask) < 100:
            X_train_amb, y_train_amb = X_train, y_train
        else:
            X_train_amb = X_train[amb_mask]
            y_train_amb = y_train[amb_mask]

        n_pos_amb = sum(y_train_amb)
        n_neg_amb = len(y_train_amb) - n_pos_amb

        # Train Stage B with strict settings
        stage_b = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=cfg["b_depth"],
            num_leaves=min(2**cfg["b_depth"] - 1, 127),
            learning_rate=0.03,
            min_child_samples=30,
            scale_pos_weight=(n_neg_amb / n_pos_amb) * cfg["b_scale"] if n_pos_amb > 0 else 1.0,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=42,
            verbose=-1,
        )
        stage_b.fit(X_train_amb, y_train_amb)

        stage_b_val_proba = stage_b.predict_proba(X_val)[:, 1]

        # Search for optimal Stage B threshold
        best_result = None
        best_score = 0

        for b_thresh in np.arange(0.20, 0.80, 0.02):
            y_pred = np.zeros(len(y_val), dtype=int)

            low_mask = stage_a_val_proba < cfg["a_low"]
            high_mask = stage_a_val_proba > cfg["a_high"]
            amb_mask_val = ~low_mask & ~high_mask

            y_pred[low_mask] = 0
            y_pred[high_mask] = 1
            y_pred[amb_mask_val] = (stage_b_val_proba[amb_mask_val] > b_thresh).astype(int)

            p = precision_score(y_val, y_pred, zero_division=0)
            r = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            # Score: prioritize meeting targets, then F1
            targets_met = p >= 0.90 and r >= 0.90
            score = (1000 if targets_met else 0) + f1

            if score > best_score:
                best_score = score
                best_result = {
                    "config": cfg["name"],
                    "stage_b_threshold": float(b_thresh),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                    "targets_met": targets_met,
                    "stage_a": stage_a,
                    "stage_b": stage_b,
                    "stage_a_low": cfg["a_low"],
                    "stage_a_high": cfg["a_high"],
                }

        if best_result:
            results.append(best_result)
            status = "*** TARGETS MET ***" if best_result["targets_met"] else ""
            print(f"  Best: P={best_result['precision']:.2%}, R={best_result['recall']:.2%}, "
                  f"F1={best_result['f1']:.4f} {status}")

    return results


def run_v3_experiments():
    """Run v3 aggressive precision experiments."""
    print("=" * 60)
    print("HARPOCRATES ML EXPERIMENTS V3 - AGGRESSIVE PRECISION")
    print("=" * 60)
    print("Target: Precision >= 90%, Recall >= 90%")
    print()

    # Load existing data
    data_path = Path("Harpocrates/training/data/training_data_v2.pkl")
    print("Loading training data...")
    X, y, records = load_training_data(data_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # Generate additional hard negatives
    print("\nGenerating additional hard negatives...")
    extra_records = generate_more_hard_negatives(count=15000)
    X_extra, y_extra, _ = load_training_data_from_records(extra_records)

    # Combine datasets
    X_combined = np.vstack([X, X_extra])
    y_combined = np.concatenate([y, y_extra])
    print(f"Combined dataset: {len(X_combined)} samples")
    print(f"Positive rate: {sum(y_combined)/len(y_combined):.1%}")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(X_combined))
    split = int(0.8 * len(X_combined))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X_combined[train_idx], y_combined[train_idx]
    X_val, y_val = X_combined[val_idx], y_combined[val_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train pos: {sum(y_train)/len(y_train):.1%}, Val pos: {sum(y_val)/len(y_val):.1%}")

    # Run experiments
    print("\n" + "=" * 60)
    print("RUNNING PRECISION-FOCUSED EXPERIMENTS")
    print("=" * 60)

    results = train_precision_focused(X_train, y_train, X_val, y_val)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)

    print(f"\n{'Config':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Target':>8}")
    print("-" * 60)

    for r in results_sorted:
        status = "YES" if r["targets_met"] else "no"
        print(f"{r['config']:<20} {r['precision']:>10.2%} {r['recall']:>10.2%} "
              f"{r['f1']:>10.4f} {status:>8}")

    # Check if any met targets
    targets_met = [r for r in results if r["targets_met"]]
    if targets_met:
        print("\n*** CONFIGURATIONS MEETING TARGETS ***")
        for r in targets_met:
            print(f"  {r['config']}: P={r['precision']:.2%}, R={r['recall']:.2%}")

    # Save best model
    best = results_sorted[0]
    output_dir = Path("Harpocrates/ml/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_a_path = output_dir / "stageA_xgboost.json"
    stage_b_path = output_dir / "stageB_lightgbm.txt"

    best["stage_a"].save_model(str(stage_a_path))
    best["stage_b"].booster_.save_model(str(stage_b_path))

    config = {
        "version": "2.3.0",
        "mode": "two_stage",
        "created_at": datetime.now().isoformat(),
        "experiment": best["config"],
        "stage_a": {
            "model_type": "xgboost",
            "features": "token_only",
            "feature_count": 23,
            "threshold_low": best["stage_a_low"],
            "threshold_high": best["stage_a_high"],
        },
        "stage_b": {
            "model_type": "lightgbm",
            "features": "all",
            "feature_count": 51,
            "threshold": best["stage_b_threshold"],
        },
        "combined_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        },
        "targets_met": best["targets_met"],
    }

    config_path = output_dir / "two_stage_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved best model: {best['config']}")
    print(f"  Precision: {best['precision']:.2%}")
    print(f"  Recall: {best['recall']:.2%}")
    print(f"  F1: {best['f1']:.4f}")


def load_training_data_from_records(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load features from records list."""
    from Harpocrates.ml.features import extract_features_from_record

    features = []
    labels = []
    valid_records = []

    for record in records:
        try:
            fv = extract_features_from_record(record)
            features.append(fv.to_array())
            labels.append(record["label"])
            valid_records.append(record)
        except Exception:
            continue

    return np.array(features), np.array(labels), valid_records


if __name__ == "__main__":
    run_v3_experiments()
