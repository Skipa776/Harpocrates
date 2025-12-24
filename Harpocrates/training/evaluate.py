"""
Model evaluation for Harpocrates ML.

Provides comprehensive evaluation metrics, confusion matrix analysis,
and error analysis for trained models.

Usage:
    python -m Harpocrates.training.evaluate \
        --model Harpocrates/ml/models/xgboost_v1.json \
        --data test_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from Harpocrates.training.dataset import Dataset


def load_model(model_path: Path) -> Tuple[Any, float]:
    """
    Load trained model and configuration.

    Args:
        model_path: Path to model JSON file

    Returns:
        Tuple of (model, threshold)
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "Evaluation requires ML dependencies. Install with: pip install harpocrates[ml]"
        ) from e

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = xgb.Booster()
    model.load_model(str(model_path))

    # Load config for threshold
    config_path = model_path.with_name("feature_config.json")
    threshold = 0.5

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            threshold = config.get("threshold", 0.5)

    return model, threshold


def evaluate_model(
    model: Any,
    dataset: Dataset,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.

    Args:
        model: Trained XGBoost Booster
        dataset: Dataset to evaluate
        threshold: Classification threshold

    Returns:
        Dict with evaluation metrics
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X = np.array(dataset.features)
    y = np.array(dataset.labels)

    # Predict probabilities
    dmatrix = xgb.DMatrix(X)
    y_proba = model.predict(dmatrix)

    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        "threshold": threshold,
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }

    # Classification report
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    # Precision-recall curve data
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    metrics["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }

    return metrics


def analyze_errors(
    model: Any,
    dataset: Dataset,
    threshold: float = 0.5,
    max_examples: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze misclassified examples.

    Args:
        model: Trained XGBoost Booster
        dataset: Dataset to analyze
        threshold: Classification threshold
        max_examples: Maximum examples per error type

    Returns:
        Dict with false_positives and false_negatives lists
    """
    import numpy as np
    import xgboost as xgb

    X = np.array(dataset.features)
    y = np.array(dataset.labels)

    # Predict
    dmatrix = xgb.DMatrix(X)
    y_proba = model.predict(dmatrix)
    y_pred = (y_proba >= threshold).astype(int)

    false_positives = []
    false_negatives = []

    for i, (true_label, pred_label, prob) in enumerate(zip(y, y_pred, y_proba)):
        record = dataset.records[i]

        if true_label == 0 and pred_label == 1:
            # False positive: predicted secret but actually safe
            false_positives.append({
                "token": record.get("token", "")[:20] + "...",
                "line_content": record.get("line_content", ""),
                "label_reason": record.get("label_reason", ""),
                "probability": float(prob),
                "negative_type": record.get("negative_type", ""),
            })
        elif true_label == 1 and pred_label == 0:
            # False negative: predicted safe but actually secret
            false_negatives.append({
                "token": record.get("token", "")[:20] + "...",
                "line_content": record.get("line_content", ""),
                "label_reason": record.get("label_reason", ""),
                "probability": float(prob),
                "secret_type": record.get("secret_type", ""),
            })

    # Sort by probability (most confident errors first)
    false_positives.sort(key=lambda x: -x["probability"])
    false_negatives.sort(key=lambda x: x["probability"])

    return {
        "false_positives": false_positives[:max_examples],
        "false_negatives": false_negatives[:max_examples],
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation metrics in formatted output."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print("\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Neg    Pos")
    print(f"  Actual Neg  {cm['true_negative']:5d}  {cm['false_positive']:5d}")
    print(f"  Actual Pos  {cm['false_negative']:5d}  {cm['true_positive']:5d}")

    report = metrics["classification_report"]
    print("\nPer-Class Metrics:")
    print(f"  Class 0 (Not Secret): P={report['0']['precision']:.3f}, "
          f"R={report['0']['recall']:.3f}, F1={report['0']['f1-score']:.3f}")
    print(f"  Class 1 (Secret):     P={report['1']['precision']:.3f}, "
          f"R={report['1']['recall']:.3f}, F1={report['1']['f1-score']:.3f}")


def print_error_analysis(errors: Dict[str, Any]) -> None:
    """Print error analysis in formatted output."""
    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)

    print(f"\nTotal False Positives: {errors['total_false_positives']}")
    print(f"Total False Negatives: {errors['total_false_negatives']}")

    if errors["false_positives"]:
        print("\nFalse Positives (predicted secret, actually safe):")
        for i, fp in enumerate(errors["false_positives"], 1):
            print(f"\n  {i}. Type: {fp.get('negative_type', 'unknown')}")
            print(f"     Prob: {fp['probability']:.3f}")
            print(f"     Line: {fp['line_content'][:60]}...")
            print(f"     Reason: {fp.get('label_reason', 'N/A')}")

    if errors["false_negatives"]:
        print("\nFalse Negatives (predicted safe, actually secret):")
        for i, fn in enumerate(errors["false_negatives"], 1):
            print(f"\n  {i}. Type: {fn.get('secret_type', 'unknown')}")
            print(f"     Prob: {fn['probability']:.3f}")
            print(f"     Line: {fn['line_content'][:60]}...")
            print(f"     Reason: {fn.get('label_reason', 'N/A')}")


def find_optimal_threshold(
    model: Any,
    dataset: Dataset,
    target_precision: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold for target precision.

    Args:
        model: Trained model
        dataset: Validation dataset
        target_precision: Target precision

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import precision_recall_curve

    X = np.array(dataset.features)
    y = np.array(dataset.labels)

    dmatrix = xgb.DMatrix(X)
    y_proba = model.predict(dmatrix)

    precision, recall, thresholds = precision_recall_curve(y, y_proba)

    # Find threshold achieving target precision with best recall
    valid_idx = np.where(precision >= target_precision)[0]

    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(recall[valid_idx])]
        optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        metrics = {
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "threshold": optimal_threshold,
        }
    else:
        # Target not achievable, use F1-optimal
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        metrics = {
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "threshold": optimal_threshold,
            "note": f"Target precision {target_precision:.0%} not achievable",
        }

    return optimal_threshold, metrics


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate Harpocrates ML model"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Path to trained model JSON file",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        required=True,
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Classification threshold (default: from config)",
    )
    parser.add_argument(
        "--error-examples",
        type=int,
        default=5,
        help="Number of error examples to show (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--find-threshold",
        action="store_true",
        help="Find optimal threshold for target precision",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="Target precision for threshold search (default: 0.95)",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, config_threshold = load_model(args.model)

    threshold = args.threshold if args.threshold is not None else config_threshold
    print(f"Using threshold: {threshold}")

    print(f"Loading test data from {args.data}...")
    dataset = Dataset.from_jsonl(args.data)
    print(dataset.summary())

    # Find optimal threshold if requested
    if args.find_threshold:
        print(f"\nFinding optimal threshold for {args.target_precision:.0%} precision...")
        opt_threshold, opt_metrics = find_optimal_threshold(
            model, dataset, args.target_precision
        )
        print(f"  Optimal threshold: {opt_threshold:.4f}")
        print(f"  Precision: {opt_metrics['precision']:.4f}")
        print(f"  Recall: {opt_metrics['recall']:.4f}")
        if "note" in opt_metrics:
            print(f"  Note: {opt_metrics['note']}")
        threshold = opt_threshold

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, dataset, threshold)
    print_metrics(metrics)

    # Error analysis
    if args.error_examples > 0:
        errors = analyze_errors(model, dataset, threshold, args.error_examples)
        print_error_analysis(errors)

    # Save results if requested
    if args.output:
        results = {
            "model_path": str(args.model),
            "data_path": str(args.data),
            "metrics": metrics,
        }
        if args.error_examples > 0:
            results["errors"] = errors

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
