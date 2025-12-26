#!/usr/bin/env python3
"""
Training with improved data generator (v4).
Uses training_data_v3.pkl which has better variable name assignment
for prefixed negative samples.
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


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train both stages and evaluate combined pipeline."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    token_indices = get_token_feature_indices()
    X_train_tokens = X_train[:, token_indices]
    X_val_tokens = X_val[:, token_indices]

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos

    # Train Stage A
    stage_a = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=config.get("a_depth", 5),
        learning_rate=0.1,
        n_estimators=config.get("a_estimators", 120),
        min_child_weight=3,
        scale_pos_weight=(n_neg / n_pos) * config.get("a_scale", 1.5),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    stage_a.fit(X_train_tokens, y_train, verbose=False)

    stage_a_train_proba = stage_a.predict_proba(X_train_tokens)[:, 1]
    stage_a_val_proba = stage_a.predict_proba(X_val_tokens)[:, 1]

    # Filter ambiguous for Stage B training
    a_low = config.get("a_low", 0.15)
    a_high = config.get("a_high", 0.85)
    amb_mask = (stage_a_train_proba >= a_low) & (stage_a_train_proba <= a_high)

    if sum(amb_mask) < 100:
        X_train_amb, y_train_amb = X_train, y_train
    else:
        X_train_amb = X_train[amb_mask]
        y_train_amb = y_train[amb_mask]

    n_pos_amb = sum(y_train_amb)
    n_neg_amb = len(y_train_amb) - n_pos_amb

    # Train Stage B
    stage_b = lgb.LGBMClassifier(
        n_estimators=config.get("b_estimators", 300),
        max_depth=config.get("b_depth", 12),
        num_leaves=config.get("b_leaves", 63),
        learning_rate=0.03,
        min_child_samples=30,
        scale_pos_weight=(n_neg_amb / n_pos_amb) * config.get("b_scale", 0.5) if n_pos_amb > 0 else 1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=42,
        verbose=-1,
    )
    stage_b.fit(X_train_amb, y_train_amb)

    stage_b_val_proba = stage_b.predict_proba(X_val)[:, 1]

    # Search for optimal Stage B threshold
    best_result = None
    best_score = 0

    for b_thresh in np.arange(0.20, 0.75, 0.02):
        y_pred = np.zeros(len(y_val), dtype=int)

        low_mask = stage_a_val_proba < a_low
        high_mask = stage_a_val_proba > a_high
        amb_mask_val = ~low_mask & ~high_mask

        y_pred[low_mask] = 0
        y_pred[high_mask] = 1
        y_pred[amb_mask_val] = (stage_b_val_proba[amb_mask_val] > b_thresh).astype(int)

        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        targets_met = p >= 0.90 and r >= 0.90
        score = (1000 if targets_met else 0) + f1

        if score > best_score:
            best_score = score
            cm = confusion_matrix(y_val, y_pred)
            best_result = {
                "config_name": config.get("name", "unnamed"),
                "stage_b_threshold": float(b_thresh),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "targets_met": targets_met,
                "stage_a": stage_a,
                "stage_b": stage_b,
                "a_low": a_low,
                "a_high": a_high,
                "confusion_matrix": cm.tolist(),
            }

    return best_result


def main():
    print("=" * 60)
    print("HARPOCRATES ML TRAINING V4 - IMPROVED DATA")
    print("=" * 60)
    print("Target: Precision >= 90%, Recall >= 90%")
    print()

    # Load improved data
    data_path = Path("Harpocrates/training/data/training_data_v3.pkl")
    print("Loading improved training data...")
    X, y, records = load_training_data(data_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train pos: {sum(y_train)/len(y_train):.1%}, Val pos: {sum(y_val)/len(y_val):.1%}")

    # Define configurations
    configs = [
        {"name": "baseline", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.5, "b_depth": 12},
        {"name": "strict_b", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.3, "b_depth": 14},
        {"name": "narrow_a", "a_low": 0.25, "a_high": 0.75, "a_scale": 1.5, "b_scale": 0.4, "b_depth": 12},
        {"name": "wide_a", "a_low": 0.10, "a_high": 0.90, "a_scale": 1.8, "b_scale": 0.4, "b_depth": 12},
        {"name": "balanced", "a_low": 0.18, "a_high": 0.82, "a_scale": 1.6, "b_scale": 0.45, "b_depth": 12},
        {"name": "deep_ensemble", "a_low": 0.15, "a_high": 0.85, "a_scale": 1.5, "b_scale": 0.35, "b_depth": 16, "b_leaves": 127},
    ]

    results = []
    best_result = None
    best_score = 0

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for cfg in configs:
        print(f"\n{cfg['name']}:")
        result = train_and_evaluate(X_train, y_train, X_val, y_val, cfg)
        if result:
            results.append(result)
            status = "*** TARGETS MET ***" if result["targets_met"] else ""
            print(f"  P={result['precision']:.2%}, R={result['recall']:.2%}, F1={result['f1']:.4f} {status}")

            score = (1000 if result["targets_met"] else 0) + result["f1"]
            if score > best_score:
                best_score = score
                best_result = result

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    print(f"\n{'Config':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Target':>8}")
    print("-" * 60)

    for r in results_sorted:
        status = "YES" if r["targets_met"] else "no"
        print(f"{r['config_name']:<20} {r['precision']:>10.2%} {r['recall']:>10.2%} "
              f"{r['f1']:>10.4f} {status:>8}")

    # Save best model
    if best_result:
        output_dir = Path("Harpocrates/ml/models")
        output_dir.mkdir(parents=True, exist_ok=True)

        stage_a_path = output_dir / "stageA_xgboost.json"
        stage_b_path = output_dir / "stageB_lightgbm.txt"

        best_result["stage_a"].save_model(str(stage_a_path))
        best_result["stage_b"].booster_.save_model(str(stage_b_path))

        config = {
            "version": "2.4.0",
            "mode": "two_stage",
            "created_at": datetime.now().isoformat(),
            "experiment": best_result["config_name"],
            "stage_a": {
                "model_type": "xgboost",
                "features": "token_only",
                "feature_count": 23,
                "threshold_low": best_result["a_low"],
                "threshold_high": best_result["a_high"],
            },
            "stage_b": {
                "model_type": "lightgbm",
                "features": "all",
                "feature_count": 51,
                "threshold": best_result["stage_b_threshold"],
            },
            "combined_metrics": {
                "precision": best_result["precision"],
                "recall": best_result["recall"],
                "f1": best_result["f1"],
            },
            "confusion_matrix": best_result["confusion_matrix"],
            "targets_met": best_result["targets_met"],
        }

        config_path = output_dir / "two_stage_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved best model: {best_result['config_name']}")
        print(f"  Precision: {best_result['precision']:.2%}")
        print(f"  Recall: {best_result['recall']:.2%}")
        print(f"  F1: {best_result['f1']:.4f}")
        print(f"  Confusion matrix: {best_result['confusion_matrix']}")


if __name__ == "__main__":
    main()
