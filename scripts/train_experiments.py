#!/usr/bin/env python3
"""
Training experiments for Harpocrates ML pipeline.

Runs multiple configurations to find optimal settings for achieving:
- Precision >= 90%
- Recall >= 90%

Usage:
    python scripts/train_experiments.py
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""

    name: str
    stage_a_threshold_low: float
    stage_a_threshold_high: float
    stage_a_max_depth: int
    stage_a_n_estimators: int
    stage_a_scale_pos_multiplier: float
    stage_b_max_depth: int
    stage_b_n_estimators: int
    stage_b_num_leaves: int
    stage_b_scale_pos_multiplier: float
    stage_b_min_recall_target: float
    seed: int = 42


def load_training_data(
    data_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load training data from pickle file."""
    import logging

    logger = logging.getLogger(__name__)
    from Harpocrates.ml.features import FeatureVector, extract_features_from_record

    with open(data_path, "rb") as f:
        records = pickle.load(f)

    features = []
    labels = []
    valid_records = []
    dropped = 0

    for i, record in enumerate(records):
        try:
            fv = extract_features_from_record(record)
            features.append(fv.to_array())
            labels.append(record["label"])
            valid_records.append(record)
        except Exception as e:
            logger.warning(
                f"Skipping record {record.get('id', i)}: "
                f"{type(e).__name__}: {e}"
            )
            dropped += 1
            continue

    if dropped > 0:
        logger.info(f"Dropped {dropped}/{len(records)} records during feature extraction")

    return np.array(features), np.array(labels), valid_records


def get_token_feature_indices() -> List[int]:
    """Get indices of token-level features (first 23 features)."""
    return list(range(23))


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Train both stages and evaluate combined pipeline."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import (
        precision_recall_curve,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    results = {"config": config.__dict__}
    token_indices = get_token_feature_indices()

    # === STAGE A: High-recall token detector ===
    X_train_tokens = X_train[:, token_indices]
    X_val_tokens = X_val[:, token_indices]

    n_positive = sum(y_train)
    n_negative = len(y_train) - n_positive

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            f"Single-class training set: {n_positive} positive, {n_negative} negative"
        )

    n_val_positive = sum(y_val)
    n_val_negative = len(y_val) - n_val_positive
    if n_val_positive == 0 or n_val_negative == 0:
        raise ValueError(
            f"Single-class validation set: {n_val_positive} positive, {n_val_negative} negative"
        )

    scale_pos_weight = (n_negative / n_positive) * config.stage_a_scale_pos_multiplier

    stage_a_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": config.stage_a_max_depth,
        "learning_rate": 0.1,
        "n_estimators": config.stage_a_n_estimators,
        "min_child_weight": 3,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": config.seed,
        "n_jobs": -1,
    }

    stage_a_model = xgb.XGBClassifier(**stage_a_params)
    stage_a_model.fit(X_train_tokens, y_train, eval_set=[(X_val_tokens, y_val)], verbose=False)

    # Get Stage A predictions
    stage_a_train_proba = stage_a_model.predict_proba(X_train_tokens)[:, 1]
    stage_a_val_proba = stage_a_model.predict_proba(X_val_tokens)[:, 1]

    # Stage A metrics
    precisions_a, recalls_a, thresholds_a = precision_recall_curve(y_val, stage_a_val_proba)
    stage_a_auc = roc_auc_score(y_val, stage_a_val_proba)

    results["stage_a"] = {
        "auc_roc": float(stage_a_auc),
        "threshold_low": config.stage_a_threshold_low,
        "threshold_high": config.stage_a_threshold_high,
    }

    # === STAGE B: High-precision context verifier ===
    # Filter to ambiguous samples
    ambiguous_mask_train = (stage_a_train_proba >= config.stage_a_threshold_low) & (
        stage_a_train_proba <= config.stage_a_threshold_high
    )

    if sum(ambiguous_mask_train) < 100:
        X_train_ambiguous = X_train
        y_train_ambiguous = y_train
    else:
        X_train_ambiguous = X_train[ambiguous_mask_train]
        y_train_ambiguous = y_train[ambiguous_mask_train]

    n_pos_amb = sum(y_train_ambiguous)
    n_neg_amb = len(y_train_ambiguous) - n_pos_amb

    if n_pos_amb == 0 or n_neg_amb == 0:
        print("Warning: single-class ambiguous subset, using fallback prediction")
        # Fallback: predict based on Stage A only
        y_pred_fallback = (stage_a_val_proba > 0.5).astype(int)
        low_mask = stage_a_val_proba < config.stage_a_threshold_low
        high_mask = stage_a_val_proba > config.stage_a_threshold_high
        amb_mask_f = ~low_mask & ~high_mask

        results["stage_a"] = {
            "auc_roc": float(stage_a_auc),
            "threshold_low": config.stage_a_threshold_low,
            "threshold_high": config.stage_a_threshold_high,
        }
        results["stage_b"] = {"threshold": 0.5, "skipped": True}
        results["combined"] = {
            "precision": float(precision_score(y_val, y_pred_fallback, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred_fallback, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred_fallback, zero_division=0)),
            "rejected_by_stage_a": int(sum(low_mask)),
            "accepted_by_stage_a": int(sum(high_mask)),
            "sent_to_stage_b": int(sum(amb_mask_f)),
        }
        results["targets_met"] = (
            results["combined"]["precision"] >= 0.90
            and results["combined"]["recall"] >= 0.90
        )
        results["models"] = {"stage_a": stage_a_model, "stage_b": None}
        results["note"] = "single-class ambiguous subset, Stage A only"
        return results

    scale_pos_weight_b = (
        (n_neg_amb / n_pos_amb) * config.stage_b_scale_pos_multiplier
    )

    stage_b_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "max_depth": config.stage_b_max_depth,
        "learning_rate": 0.05,
        "n_estimators": config.stage_b_n_estimators,
        "num_leaves": config.stage_b_num_leaves,
        "min_child_samples": 20,
        "scale_pos_weight": scale_pos_weight_b,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": config.seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    stage_b_model = lgb.LGBMClassifier(**stage_b_params)
    stage_b_model.fit(X_train_ambiguous, y_train_ambiguous)

    stage_b_val_proba = stage_b_model.predict_proba(X_val)[:, 1]

    # Find optimal Stage B threshold for target recall
    precisions_b, recalls_b, thresholds_b = precision_recall_curve(y_val, stage_b_val_proba)

    # Try multiple thresholds and pick best
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_b = (stage_b_val_proba > thresh).astype(int)
        if sum(y_pred_b) == 0:
            continue
        p = precision_score(y_val, y_pred_b, zero_division=0)
        r = recall_score(y_val, y_pred_b, zero_division=0)
        f1 = 2 * p * r / (p + r + 1e-10)
        if r >= config.stage_b_min_recall_target and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    results["stage_b"] = {
        "threshold": float(best_threshold),
        "ambiguous_samples": int(sum(ambiguous_mask_train)),
    }

    # === COMBINED EVALUATION ===
    y_pred = np.zeros(len(y_val), dtype=int)

    # Low confidence → Negative
    low_mask = stage_a_val_proba < config.stage_a_threshold_low
    y_pred[low_mask] = 0

    # High confidence → Positive
    high_mask = stage_a_val_proba > config.stage_a_threshold_high
    y_pred[high_mask] = 1

    # Ambiguous → Use Stage B
    ambiguous_mask = ~low_mask & ~high_mask
    y_pred[ambiguous_mask] = (stage_b_val_proba[ambiguous_mask] > best_threshold).astype(int)

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    results["combined"] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "rejected_by_stage_a": int(sum(low_mask)),
        "accepted_by_stage_a": int(sum(high_mask)),
        "sent_to_stage_b": int(sum(ambiguous_mask)),
    }

    # Check if targets met
    results["targets_met"] = precision >= 0.90 and recall >= 0.90
    results["models"] = {"stage_a": stage_a_model, "stage_b": stage_b_model}

    return results


def run_experiments() -> List[Dict[str, Any]]:
    """Run all experiment configurations."""
    print("=" * 60)
    print("HARPOCRATES ML TRAINING EXPERIMENTS")
    print("=" * 60)
    print(f"Target: Precision >= 90%, Recall >= 90%")
    print()

    # Load data
    data_path = Path("Harpocrates/training/data/training_data_v2.pkl")
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        return []

    print("Loading training data...")
    X, y, records = load_training_data(data_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # Split into train/val (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Train positive rate: {sum(y_train)/len(y_train):.1%}")
    print(f"Val positive rate: {sum(y_val)/len(y_val):.1%}")

    # Define experiment configurations
    experiments = [
        # Baseline
        ExperimentConfig(
            name="baseline",
            stage_a_threshold_low=0.10,
            stage_a_threshold_high=0.90,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=8,
            stage_b_n_estimators=200,
            stage_b_num_leaves=31,
            stage_b_scale_pos_multiplier=0.8,
            stage_b_min_recall_target=0.85,
        ),
        # Tighter Stage A window (more to Stage B)
        ExperimentConfig(
            name="tight_stage_a",
            stage_a_threshold_low=0.20,
            stage_a_threshold_high=0.80,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=8,
            stage_b_n_estimators=200,
            stage_b_num_leaves=31,
            stage_b_scale_pos_multiplier=0.8,
            stage_b_min_recall_target=0.85,
        ),
        # Very tight Stage A (most to Stage B)
        ExperimentConfig(
            name="very_tight_stage_a",
            stage_a_threshold_low=0.30,
            stage_a_threshold_high=0.70,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=8,
            stage_b_n_estimators=200,
            stage_b_num_leaves=31,
            stage_b_scale_pos_multiplier=0.8,
            stage_b_min_recall_target=0.85,
        ),
        # Deeper Stage B
        ExperimentConfig(
            name="deep_stage_b",
            stage_a_threshold_low=0.15,
            stage_a_threshold_high=0.85,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=12,
            stage_b_n_estimators=300,
            stage_b_num_leaves=63,
            stage_b_scale_pos_multiplier=0.8,
            stage_b_min_recall_target=0.85,
        ),
        # Higher Stage A recall (more scale_pos)
        ExperimentConfig(
            name="high_recall_stage_a",
            stage_a_threshold_low=0.05,
            stage_a_threshold_high=0.95,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=2.0,
            stage_b_max_depth=8,
            stage_b_n_estimators=200,
            stage_b_num_leaves=31,
            stage_b_scale_pos_multiplier=0.8,
            stage_b_min_recall_target=0.90,
        ),
        # Balanced Stage B
        ExperimentConfig(
            name="balanced_stage_b",
            stage_a_threshold_low=0.15,
            stage_a_threshold_high=0.85,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=8,
            stage_b_n_estimators=200,
            stage_b_num_leaves=31,
            stage_b_scale_pos_multiplier=1.0,  # Balanced
            stage_b_min_recall_target=0.90,
        ),
        # High precision Stage B
        ExperimentConfig(
            name="high_precision_stage_b",
            stage_a_threshold_low=0.15,
            stage_a_threshold_high=0.85,
            stage_a_max_depth=4,
            stage_a_n_estimators=100,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=10,
            stage_b_n_estimators=250,
            stage_b_num_leaves=47,
            stage_b_scale_pos_multiplier=0.6,  # More conservative
            stage_b_min_recall_target=0.88,
        ),
        # Deeper Stage A
        ExperimentConfig(
            name="deep_stage_a",
            stage_a_threshold_low=0.15,
            stage_a_threshold_high=0.85,
            stage_a_max_depth=6,
            stage_a_n_estimators=150,
            stage_a_scale_pos_multiplier=1.5,
            stage_b_max_depth=8,
            stage_b_n_estimators=200,
            stage_b_num_leaves=31,
            stage_b_scale_pos_multiplier=0.8,
            stage_b_min_recall_target=0.85,
        ),
        # Aggressive thresholds for high precision
        ExperimentConfig(
            name="aggressive_precision",
            stage_a_threshold_low=0.25,
            stage_a_threshold_high=0.75,
            stage_a_max_depth=5,
            stage_a_n_estimators=120,
            stage_a_scale_pos_multiplier=1.3,
            stage_b_max_depth=10,
            stage_b_n_estimators=250,
            stage_b_num_leaves=47,
            stage_b_scale_pos_multiplier=0.7,
            stage_b_min_recall_target=0.90,
        ),
        # Combined optimizations
        ExperimentConfig(
            name="combined_optimal",
            stage_a_threshold_low=0.12,
            stage_a_threshold_high=0.88,
            stage_a_max_depth=5,
            stage_a_n_estimators=120,
            stage_a_scale_pos_multiplier=1.8,
            stage_b_max_depth=10,
            stage_b_n_estimators=250,
            stage_b_num_leaves=47,
            stage_b_scale_pos_multiplier=0.9,
            stage_b_min_recall_target=0.90,
        ),
    ]

    results = []
    best_result = None
    best_score = 0

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENTS")
    print("=" * 60)

    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {config.name}")
        print("-" * 40)

        try:
            result = train_and_evaluate(X_train, y_train, X_val, y_val, config)

            precision = result["combined"]["precision"]
            recall = result["combined"]["recall"]
            f1 = result["combined"]["f1"]

            print(f"  Precision: {precision:.2%}")
            print(f"  Recall:    {recall:.2%}")
            print(f"  F1:        {f1:.4f}")
            print(f"  Stage A routing: {result['combined']['rejected_by_stage_a']} rejected, "
                  f"{result['combined']['accepted_by_stage_a']} accepted, "
                  f"{result['combined']['sent_to_stage_b']} to Stage B")

            if result["targets_met"]:
                print(f"  *** TARGETS MET! ***")

            # Track best result
            # Score: prioritize meeting targets, then F1
            score = (1000 if result["targets_met"] else 0) + f1
            if score > best_score:
                best_score = score
                best_result = result

            # Remove models from results for serialization
            result_copy = {k: v for k, v in result.items() if k != "models"}
            results.append(result_copy)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Sort by F1
    sorted_results = sorted(results, key=lambda x: x["combined"]["f1"], reverse=True)

    print("\nTop 5 configurations by F1:")
    print("-" * 60)
    print(f"{'Name':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Target':>8}")
    print("-" * 60)

    for result in sorted_results[:5]:
        name = result["config"]["name"]
        p = result["combined"]["precision"]
        r = result["combined"]["recall"]
        f1 = result["combined"]["f1"]
        target = "YES" if result["targets_met"] else "no"
        print(f"{name:<25} {p:>10.2%} {r:>10.2%} {f1:>10.4f} {target:>8}")

    # Save best model
    if best_result:
        print("\n" + "=" * 60)
        print("SAVING BEST MODEL")
        print("=" * 60)

        output_dir = Path("Harpocrates/ml/models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        stage_a_path = output_dir / "stageA_xgboost.json"
        stage_b_path = output_dir / "stageB_lightgbm.txt"

        if best_result["models"]["stage_a"] is not None:
            best_result["models"]["stage_a"].save_model(str(stage_a_path))
        else:
            print("Warning: Stage A model is None, skipping save")

        if best_result["models"]["stage_b"] is not None and hasattr(best_result["models"]["stage_b"], "booster_"):
            best_result["models"]["stage_b"].booster_.save_model(str(stage_b_path))
        else:
            print("Warning: Stage B model is None or has no booster_, skipping save")

        # Save config
        config = {
            "version": "2.1.0",
            "mode": "two_stage",
            "created_at": datetime.now().isoformat(),
            "experiment_name": best_result["config"]["name"],
            "stage_a": {
                "model_type": "xgboost",
                "features": "token_only",
                "feature_count": X_train[:, :len(get_token_feature_indices())].shape[1],
                "threshold_low": best_result["config"]["stage_a_threshold_low"],
                "threshold_high": best_result["config"]["stage_a_threshold_high"],
                "metrics": best_result["stage_a"],
            },
            "stage_b": {
                "model_type": "lightgbm",
                "features": "all",
                "feature_count": X_train.shape[1],
                "threshold": best_result["stage_b"]["threshold"],
            },
            "combined_metrics": best_result["combined"],
        }

        config_path = output_dir / "two_stage_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved Stage A: {stage_a_path}")
        print(f"Saved Stage B: {stage_b_path}")
        print(f"Saved config: {config_path}")

        print(f"\nBest model: {best_result['config']['name']}")
        print(f"  Precision: {best_result['combined']['precision']:.2%}")
        print(f"  Recall: {best_result['combined']['recall']:.2%}")
        print(f"  F1: {best_result['combined']['f1']:.4f}")

    # Save all results
    results_path = Path("Harpocrates/training/data/experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nAll results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_experiments()
