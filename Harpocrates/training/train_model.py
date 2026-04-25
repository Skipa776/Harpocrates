"""
Single-stage XGBoost training for Harpocrates ML.

Trains a single XGBoost model on all 65 features with Platt scaling
for calibrated probabilities and dual-threshold decision logic.

Usage:
    python -m Harpocrates.training.train_model \
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


N_FEATURES = 65


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Load dataset and extract 65-feature vectors from JSONL records."""
    from Harpocrates.ml.features import extract_features_from_record

    records: List[dict] = []
    features: List[List[float]] = []
    labels: List[int] = []

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
            try:
                fv = extract_features_from_record(record)
                arr = fv.to_array()
                if len(arr) != N_FEATURES:
                    print(
                        f"Warning: Expected {N_FEATURES} features, got {len(arr)}. "
                        f"Skipping record."
                    )
                    continue
                features.append(arr)
                labels.append(record["label"])
            except Exception as e:
                print(f"Warning: Failed to extract features: {e}")
                continue

    return np.array(features), np.array(labels), records


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """Train XGBoost on all 65 features.

    No scale_pos_weight — class balance is enforced in data generation.
    Returns (model, metrics_dict).
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    n_positive = int(sum(y_train))
    n_negative = len(y_train) - n_positive

    if verbose:
        print("\n=== TRAINING: XGBoost on 65 features ===")
        print(f"Training samples: {len(y_train)}")
        print(f"Class balance: {n_positive} positive, {n_negative} negative")
        print(f"Positive ratio: {n_positive / len(y_train):.1%}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
    }

    model = xgb.XGBClassifier(**params)

    if X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )
    else:
        del params["early_stopping_rounds"]
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

    metrics: Dict[str, float] = {}
    if X_val is not None:
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_val, y_proba))
        if verbose:
            print(f"\nRaw XGBoost AUC-ROC: {metrics['auc_roc']:.4f}")

    return model, metrics


def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> Any:
    """Fit Platt scaling on validation set for calibrated probabilities.

    After calibration, P=0.85 means "85% of tokens with this score are
    actually secrets."
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)

    if verbose:
        raw_proba = model.predict_proba(X_val)[:, 1]
        cal_proba = calibrator.predict_proba(X_val)[:, 1]
        raw_brier = brier_score_loss(y_val, raw_proba)
        cal_brier = brier_score_loss(y_val, cal_proba)
        print(f"\n=== PLATT SCALING ===")
        print(f"Brier score (raw):        {raw_brier:.4f}")
        print(f"Brier score (calibrated): {cal_brier:.4f}")
        if cal_brier > 0.1:
            print(
                "WARNING: Calibrated Brier score > 0.1. "
                "Fixed thresholds may not be meaningful — "
                "search empirically on calibrated outputs."
            )

    return calibrator


def find_dual_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.98,
    target_precision: float = 0.95,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Find dual thresholds on calibrated probabilities.

    Returns (threshold_low, threshold_high, optimal_single_threshold).
    - threshold_low: below this → SAFE (recall >= target_recall)
    - threshold_high: above this → SECRET (precision >= target_precision)
    - optimal_single_threshold: F1-maximizing threshold
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # threshold_low: find value where recall >= target_recall
    valid_recall = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_recall) > 0:
        best_idx = valid_recall[np.argmax(precisions[:-1][valid_recall])]
        threshold_low = float(thresholds[best_idx])
    else:
        threshold_low = 0.15

    # threshold_high: find value where precision >= target_precision
    valid_precision = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid_precision) > 0:
        best_idx = valid_precision[np.argmax(recalls[:-1][valid_precision])]
        threshold_high = float(thresholds[best_idx])
    else:
        threshold_high = 0.85

    # Ensure low < high with minimum gap
    if threshold_low >= threshold_high:
        threshold_low = 0.15
        threshold_high = 0.85

    # F1-maximizing single threshold
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-10
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])

    if verbose:
        print(f"\n=== DUAL THRESHOLDS ===")
        print(f"Threshold low  (recall >= {target_recall:.0%}):    {threshold_low:.4f}")
        print(f"Threshold high (precision >= {target_precision:.0%}): {threshold_high:.4f}")
        print(f"Optimal F1 threshold:                    {optimal_threshold:.4f}")

        # Review zone stats
        review_mask = (y_proba >= threshold_low) & (y_proba <= threshold_high)
        review_pct = review_mask.sum() / len(y_proba) * 100
        print(f"Review zone:  {review_mask.sum()}/{len(y_proba)} ({review_pct:.1f}%)")

    return threshold_low, threshold_high, optimal_threshold


def evaluate(
    calibrator: Any,
    threshold_low: float,
    threshold_high: float,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate model with dual-threshold decision logic."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_proba = calibrator.predict_proba(X_val)[:, 1]

    # Apply dual thresholds
    safe_mask = y_proba < threshold_low
    secret_mask = y_proba > threshold_high
    review_mask = ~safe_mask & ~secret_mask

    # For evaluation: REVIEW and SECRET both count as "flagged"
    y_pred = np.where(safe_mask, 0, 1)

    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_proba)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc_roc": float(auc_roc),
        "threshold_low": float(threshold_low),
        "threshold_high": float(threshold_high),
        "safe_count": int(safe_mask.sum()),
        "secret_count": int(secret_mask.sum()),
        "review_count": int(review_mask.sum()),
        "review_zone_pct": float(review_mask.sum() / len(y_val) * 100),
    }

    if verbose:
        print("\n=== EVALUATION RESULTS ===")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"AUC-ROC:   {auc_roc:.4f}")
        print(f"\nRouting breakdown:")
        print(f"  SAFE:    {safe_mask.sum()} ({safe_mask.sum()/len(y_val):.1%})")
        print(f"  REVIEW:  {review_mask.sum()} ({review_mask.sum()/len(y_val):.1%})")
        print(f"  SECRET:  {secret_mask.sum()} ({secret_mask.sum()/len(y_val):.1%})")
        print(
            f"\n{classification_report(y_val, y_pred, target_names=['Non-Secret', 'Secret'])}"
        )
        cm = confusion_matrix(y_val, y_pred)
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
        print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    return metrics


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_model(
    model: Any,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Save trained XGBoost model and configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.json"
    model.save_model(str(model_path))

    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(_convert_to_serializable(config), f, indent=2)

    if verbose:
        print(f"\nModel saved:")
        print(f"  XGBoost: {model_path}")
        print(f"  Config:  {config_path}")

    return {"model": model_path, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train single-stage XGBoost model for secret detection"
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
        help="Path to validation data JSONL file (required for calibration)",
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
        "--target-recall",
        type=float,
        default=0.98,
        help="Minimum recall for threshold_low",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="Minimum precision for threshold_high",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    try:
        import xgboost  # noqa: F401
    except ImportError as e:
        print(f"Error: Missing ML dependency. Install with: pip install harpocrates[ml]")
        print(f"Details: {e}")
        sys.exit(1)

    # Load data
    if verbose:
        print(f"Loading training data from {args.train_data}...")
    X_train, y_train, _ = load_dataset(args.train_data)
    if verbose:
        print(f"Loaded {len(X_train)} training samples ({N_FEATURES} features)")

    X_val, y_val = None, None
    if args.val_data:
        if verbose:
            print(f"Loading validation data from {args.val_data}...")
        X_val, y_val, _ = load_dataset(args.val_data)
        if verbose:
            print(f"Loaded {len(X_val)} validation samples")

    # Train XGBoost on all 65 features
    model, raw_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, seed=args.seed, verbose=verbose
    )

    # Calibrate and evaluate
    threshold_low = 0.15
    threshold_high = 0.85
    optimal_threshold = 0.5
    eval_metrics: Dict[str, Any] = {}

    if X_val is not None and y_val is not None:
        calibrator = calibrate_model(model, X_val, y_val, verbose=verbose)
        cal_proba = calibrator.predict_proba(X_val)[:, 1]

        threshold_low, threshold_high, optimal_threshold = find_dual_thresholds(
            y_val,
            cal_proba,
            target_recall=args.target_recall,
            target_precision=args.target_precision,
            verbose=verbose,
        )

        eval_metrics = evaluate(
            calibrator, threshold_low, threshold_high, X_val, y_val, verbose=verbose
        )

        # Save the calibrated model's underlying XGBoost (for ONNX conversion)
        # The Platt sigmoid params are stored in config for inference-time application
        platt_estimator = calibrator.calibrated_classifiers_[0]
        platt_a = float(platt_estimator.calibrators[0].a_)
        platt_b = float(platt_estimator.calibrators[0].b_)
    else:
        platt_a = 0.0
        platt_b = 0.0
        if verbose:
            print(
                "\nWARNING: No validation data — skipping calibration and "
                "threshold search. Using default thresholds."
            )

    config = {
        "version": "2.1.0",
        "mode": "single_stage",
        "model_type": "xgboost",
        "feature_count": N_FEATURES,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "optimal_threshold": optimal_threshold,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "metrics": eval_metrics if eval_metrics else raw_metrics,
        "training_samples": len(X_train),
        "validation_samples": len(X_val) if X_val is not None else 0,
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
    }

    save_model(model, config, args.output_dir, verbose=verbose)

    if verbose:
        print("\n=== TRAINING COMPLETE ===")
        if eval_metrics:
            print(f"Precision:       {eval_metrics['precision']:.2%}")
            print(f"Recall:          {eval_metrics['recall']:.2%}")
            print(f"F1:              {eval_metrics['f1']:.4f}")
            print(f"AUC-ROC:         {eval_metrics['auc_roc']:.4f}")
            print(f"Review zone:     {eval_metrics['review_zone_pct']:.1f}%")
            print(f"Threshold low:   {threshold_low:.4f}")
            print(f"Threshold high:  {threshold_high:.4f}")


if __name__ == "__main__":
    main()
