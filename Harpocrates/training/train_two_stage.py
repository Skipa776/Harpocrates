"""
Two-stage model training for Harpocrates ML.

Implements a two-stage detection pipeline:
- Stage A: High-recall token detector (XGBoost) - filters obvious non-secrets
- Stage B: High-precision context verifier (LightGBM) - verifies ambiguous cases

Usage:
    python -m Harpocrates.training.train_two_stage \
        --train-data train_transformed.jsonl \
        --val-data test_holdout.jsonl \
        --output-dir Harpocrates/ml/models \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load dataset and extract features."""
    from Harpocrates.ml.features import FeatureVector, extract_features_from_record

    records = []
    features = []
    labels = []

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
            try:
                fv = extract_features_from_record(record)
                features.append(fv.to_array())
                labels.append(record["label"])
            except Exception as e:
                print(f"Warning: Failed to extract features: {e}")
                continue

    return np.array(features), np.array(labels), records


def get_token_feature_indices() -> List[int]:
    """Get indices of token-level features (first 23 features).

    Token features include structural analysis but exclude context features.
    These are the features that can be computed from the token alone.
    """
    # Token features are indices 0-22 (23 features)
    # These include: token_length, entropy, char counts, base64/hex detection,
    # structure score, version pattern, and the new discriminative features
    # (is_uuid_v4, is_known_hash_length, jwt_structure_valid, entropy_charset_mismatch, has_hash_prefix)
    return list(range(23))


def train_stage_a(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Any, float, Dict[str, float]]:
    """
    Train Stage A: High-recall token detector using XGBoost.

    Uses only token-level features (18 features) for fast filtering.
    Optimized for high recall (>98%) to minimize missed secrets.
    """
    import xgboost as xgb
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    # Use only token features
    token_indices = get_token_feature_indices()
    X_train_tokens = X_train[:, token_indices]
    X_val_tokens = X_val[:, token_indices] if X_val is not None else None

    # Handle class imbalance
    n_positive = sum(y_train)
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    if verbose:
        print("\n=== STAGE A: HIGH-RECALL TOKEN DETECTOR ===")
        print(f"Training samples: {len(y_train)}")
        print(f"Using {len(token_indices)} token features (with discriminative features)")
        print(f"Class balance: {n_positive} positive, {n_negative} negative")

    # XGBoost parameters optimized for HIGH RECALL
    # - Shallow trees for speed
    # - High scale_pos_weight to catch more positives
    # - Lower learning rate for stability
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": 4,  # Shallow for speed
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_weight": 3,
        "scale_pos_weight": scale_pos_weight * 1.5,  # Bias toward positives
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping if validation data available
    if X_val_tokens is not None:
        model.fit(
            X_train_tokens,
            y_train,
            eval_set=[(X_val_tokens, y_val)],
            verbose=verbose,
        )
    else:
        model.fit(X_train_tokens, y_train)

    # Evaluate on validation set
    if X_val_tokens is not None:
        y_proba = model.predict_proba(X_val_tokens)[:, 1]

        # Find threshold for 98% recall
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        target_recall = 0.98

        # Find threshold that achieves target recall
        valid_indices = np.where(recalls[:-1] >= target_recall)[0]
        if len(valid_indices) > 0:
            # Choose threshold with best precision at target recall
            best_idx = valid_indices[np.argmax(precisions[:-1][valid_indices])]
            optimal_threshold = thresholds[best_idx]
            optimal_precision = precisions[best_idx]
            optimal_recall = recalls[best_idx]
        else:
            # Can't achieve target, use lowest threshold
            optimal_threshold = 0.1
            idx = np.argmin(np.abs(thresholds - 0.1))
            optimal_precision = precisions[idx]
            optimal_recall = recalls[idx]

        auc_roc = roc_auc_score(y_val, y_proba)

        metrics = {
            "precision": float(optimal_precision),
            "recall": float(optimal_recall),
            "threshold": float(optimal_threshold),
            "auc_roc": float(auc_roc),
        }

        if verbose:
            print(f"\nStage A Results:")
            print(f"  Threshold: {optimal_threshold:.4f}")
            print(f"  Precision: {optimal_precision:.2%}")
            print(f"  Recall: {optimal_recall:.2%}")
            print(f"  AUC-ROC: {auc_roc:.4f}")
    else:
        optimal_threshold = 0.1
        metrics = {"threshold": optimal_threshold}

    return model, optimal_threshold, metrics


def train_stage_b(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    stage_a_model: Any,
    stage_a_threshold_low: float,
    stage_a_threshold_high: float,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Any, float, Dict[str, float]]:
    """
    Train Stage B: High-precision context verifier using LightGBM.

    Uses all 46 features and only trains on ambiguous cases
    (samples where Stage A probability is between thresholds).
    Optimized for high precision (>90%).
    """
    import lightgbm as lgb
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    # Get Stage A predictions to identify ambiguous samples
    token_indices = get_token_feature_indices()
    X_train_tokens = X_train[:, token_indices]

    stage_a_proba = stage_a_model.predict_proba(X_train_tokens)[:, 1]

    # Filter to ambiguous samples only
    ambiguous_mask = (stage_a_proba >= stage_a_threshold_low) & (
        stage_a_proba <= stage_a_threshold_high
    )
    X_train_ambiguous = X_train[ambiguous_mask]
    y_train_ambiguous = y_train[ambiguous_mask]

    if verbose:
        print("\n=== STAGE B: HIGH-PRECISION CONTEXT VERIFIER ===")
        print(f"Ambiguous samples: {len(y_train_ambiguous)} / {len(y_train)}")
        print(f"Using all 51 features (including 5 new discriminative features)")

    if len(X_train_ambiguous) < 100:
        if verbose:
            print("Warning: Too few ambiguous samples, using all training data")
        X_train_ambiguous = X_train
        y_train_ambiguous = y_train

    # Handle class imbalance in ambiguous set
    n_positive = sum(y_train_ambiguous)
    n_negative = len(y_train_ambiguous) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    if verbose:
        print(f"Class balance: {n_positive} positive, {n_negative} negative")

    # LightGBM parameters optimized for HIGH PRECISION
    # - Deeper trees for complex patterns
    # - Lower scale_pos_weight to reduce false positives
    # - Strong regularization
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": scale_pos_weight * 0.8,  # Bias toward negatives
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)

    # Train
    if X_val is not None:
        # Filter validation to ambiguous samples too
        X_val_tokens = X_val[:, token_indices]
        stage_a_val_proba = stage_a_model.predict_proba(X_val_tokens)[:, 1]
        val_ambiguous_mask = (stage_a_val_proba >= stage_a_threshold_low) & (
            stage_a_val_proba <= stage_a_threshold_high
        )

        if sum(val_ambiguous_mask) > 50:
            X_val_ambiguous = X_val[val_ambiguous_mask]
            y_val_ambiguous = y_val[val_ambiguous_mask]
        else:
            X_val_ambiguous = X_val
            y_val_ambiguous = y_val

        model.fit(
            X_train_ambiguous,
            y_train_ambiguous,
            eval_set=[(X_val_ambiguous, y_val_ambiguous)],
        )
    else:
        model.fit(X_train_ambiguous, y_train_ambiguous)

    # Evaluate on validation set (full set, not just ambiguous)
    if X_val is not None:
        y_proba = model.predict_proba(X_val)[:, 1]

        # Find threshold that maximizes F1 score (balances precision and recall)
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        # Calculate F1 for each threshold
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-10
        )

        # Find threshold with best F1 that achieves at least 85% recall
        # This prioritizes catching secrets while maintaining good precision
        valid_indices = np.where(recalls[:-1] >= 0.85)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
        else:
            # Fallback to best F1 overall
            best_idx = np.argmax(f1_scores)

        optimal_threshold = thresholds[best_idx]
        optimal_precision = precisions[best_idx]
        optimal_recall = recalls[best_idx]

        auc_roc = roc_auc_score(y_val, y_proba)

        metrics = {
            "precision": float(optimal_precision),
            "recall": float(optimal_recall),
            "threshold": float(optimal_threshold),
            "f1": float(f1_scores[best_idx]),
            "auc_roc": float(auc_roc),
        }

        if verbose:
            print(f"\nStage B Results:")
            print(f"  Threshold: {optimal_threshold:.4f}")
            print(f"  Precision: {optimal_precision:.2%}")
            print(f"  Recall: {optimal_recall:.2%}")
            print(f"  AUC-ROC: {auc_roc:.4f}")
    else:
        optimal_threshold = 0.5
        metrics = {"threshold": optimal_threshold}

    return model, optimal_threshold, metrics


def evaluate_two_stage(
    stage_a_model: Any,
    stage_b_model: Any,
    stage_a_threshold_low: float,
    stage_a_threshold_high: float,
    stage_b_threshold: float,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the combined two-stage pipeline.

    Decision logic:
    - If Stage A prob < threshold_low → Negative
    - If Stage A prob > threshold_high → Positive
    - Otherwise → Use Stage B prediction
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    token_indices = get_token_feature_indices()
    X_val_tokens = X_val[:, token_indices]

    # Stage A predictions
    stage_a_proba = stage_a_model.predict_proba(X_val_tokens)[:, 1]

    # Stage B predictions (on full features)
    stage_b_proba = stage_b_model.predict_proba(X_val)[:, 1]

    # Combined predictions using decision logic
    y_pred = np.zeros(len(y_val), dtype=int)

    # Low confidence → Negative
    low_mask = stage_a_proba < stage_a_threshold_low
    y_pred[low_mask] = 0

    # High confidence → Positive
    high_mask = stage_a_proba > stage_a_threshold_high
    y_pred[high_mask] = 1

    # Ambiguous → Use Stage B
    ambiguous_mask = ~low_mask & ~high_mask
    y_pred[ambiguous_mask] = (stage_b_proba[ambiguous_mask] > stage_b_threshold).astype(
        int
    )

    # Calculate metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "stage_a_threshold_low": float(stage_a_threshold_low),
        "stage_a_threshold_high": float(stage_a_threshold_high),
        "stage_b_threshold": float(stage_b_threshold),
        "samples_rejected_by_stage_a": int(sum(low_mask)),
        "samples_accepted_by_stage_a": int(sum(high_mask)),
        "samples_sent_to_stage_b": int(sum(ambiguous_mask)),
    }

    if verbose:
        print("\n=== COMBINED TWO-STAGE RESULTS ===")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nRouting breakdown:")
        print(f"  Rejected by Stage A: {sum(low_mask)} ({sum(low_mask)/len(y_val):.1%})")
        print(f"  Accepted by Stage A: {sum(high_mask)} ({sum(high_mask)/len(y_val):.1%})")
        print(f"  Sent to Stage B: {sum(ambiguous_mask)} ({sum(ambiguous_mask)/len(y_val):.1%})")

        print(f"\n{classification_report(y_val, y_pred, target_names=['Non-Secret', 'Secret'])}")

        cm = confusion_matrix(y_val, y_pred)
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    return metrics


def save_models(
    stage_a_model: Any,
    stage_b_model: Any,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Save trained models and configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Stage A (XGBoost)
    stage_a_path = output_dir / "stageA_xgboost.json"
    stage_a_model.save_model(str(stage_a_path))

    # Save Stage B (LightGBM)
    stage_b_path = output_dir / "stageB_lightgbm.txt"
    stage_b_model.booster_.save_model(str(stage_b_path))

    # Save configuration (convert numpy types to native Python)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    config_path = output_dir / "two_stage_config.json"
    with open(config_path, "w") as f:
        json.dump(convert_to_serializable(config), f, indent=2)

    if verbose:
        print(f"\nModels saved:")
        print(f"  Stage A: {stage_a_path}")
        print(f"  Stage B: {stage_b_path}")
        print(f"  Config: {config_path}")

    return {
        "stage_a": stage_a_path,
        "stage_b": stage_b_path,
        "config": config_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train two-stage ML model for secret detection"
    )
    parser.add_argument(
        "--train-data",
        "-t",
        type=Path,
        required=True,
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--val-data",
        "-v",
        type=Path,
        help="Path to validation data JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("Harpocrates/ml/models"),
        help="Output directory for models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--stage-a-threshold-low",
        type=float,
        default=0.1,
        help="Low threshold for Stage A (below = reject)",
    )
    parser.add_argument(
        "--stage-a-threshold-high",
        type=float,
        default=0.9,
        help="High threshold for Stage A (above = accept)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Check dependencies
    try:
        import lightgbm
        import xgboost
    except ImportError as e:
        print(f"Error: Missing ML dependencies. Install with: pip install harpocrates[ml]")
        print(f"Details: {e}")
        sys.exit(1)

    # Load data
    if verbose:
        print(f"Loading training data from {args.train_data}...")
    X_train, y_train, train_records = load_dataset(args.train_data)
    if verbose:
        print(f"Loaded {len(X_train)} training samples")

    X_val, y_val = None, None
    if args.val_data:
        if verbose:
            print(f"Loading validation data from {args.val_data}...")
        X_val, y_val, _ = load_dataset(args.val_data)
        if verbose:
            print(f"Loaded {len(X_val)} validation samples")

    # Train Stage A
    stage_a_model, stage_a_optimal_threshold, stage_a_metrics = train_stage_a(
        X_train, y_train, X_val, y_val, seed=args.seed, verbose=verbose
    )

    # Train Stage B
    stage_b_model, stage_b_threshold, stage_b_metrics = train_stage_b(
        X_train,
        y_train,
        X_val,
        y_val,
        stage_a_model,
        args.stage_a_threshold_low,
        args.stage_a_threshold_high,
        seed=args.seed,
        verbose=verbose,
    )

    # Evaluate combined pipeline
    combined_metrics = {}
    if X_val is not None:
        combined_metrics = evaluate_two_stage(
            stage_a_model,
            stage_b_model,
            args.stage_a_threshold_low,
            args.stage_a_threshold_high,
            stage_b_threshold,
            X_val,
            y_val,
            verbose=verbose,
        )

    # Create configuration
    config = {
        "version": "2.0.0",
        "mode": "two_stage",
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "stage_a": {
            "model_type": "xgboost",
            "features": "token_only",
            "feature_count": 23,
            "threshold_low": args.stage_a_threshold_low,
            "threshold_high": args.stage_a_threshold_high,
            "optimal_threshold": stage_a_optimal_threshold,
            "metrics": stage_a_metrics,
        },
        "stage_b": {
            "model_type": "lightgbm",
            "features": "all",
            "feature_count": 51,
            "threshold": stage_b_threshold,
            "metrics": stage_b_metrics,
        },
        "combined_metrics": combined_metrics,
        "training_samples": len(X_train),
        "validation_samples": len(X_val) if X_val is not None else 0,
    }

    # Save models
    paths = save_models(
        stage_a_model, stage_b_model, config, args.output_dir, verbose=verbose
    )

    if verbose:
        print("\n=== TRAINING COMPLETE ===")
        if combined_metrics:
            print(f"Combined Precision: {combined_metrics['precision']:.2%}")
            print(f"Combined Recall: {combined_metrics['recall']:.2%}")
            print(f"Combined F1: {combined_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
