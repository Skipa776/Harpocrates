"""
Model training script for Harpocrates ML.

Trains binary classifiers (XGBoost, LightGBM, or ensemble) to distinguish
true secrets from false positives using contextual features.

Usage:
    # Train XGBoost (default)
    python -m Harpocrates.training.train --data training_data.jsonl

    # Train LightGBM
    python -m Harpocrates.training.train --data training_data.jsonl --model-type lightgbm

    # Train ensemble (both XGBoost and LightGBM)
    python -m Harpocrates.training.train --data training_data.jsonl --model-type ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Harpocrates.ml.features import FeatureVector
from Harpocrates.training.dataset import Dataset

# Model type options
MODEL_TYPES = ["xgboost", "lightgbm", "ensemble"]


def train_model(
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    target_precision: float = 0.95,
    verbose: bool = True,
) -> Tuple[Any, float, Dict[str, float]]:
    """
    Train XGBoost model.

    Args:
        train_data: Training dataset
        val_data: Optional validation dataset
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        n_estimators: Number of trees
        target_precision: Target precision for threshold tuning
        verbose: If True, print progress

    Returns:
        Tuple of (model, optimal_threshold, metrics)
    """
    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import (
            classification_report,
            precision_recall_curve,
            roc_auc_score,
        )
    except ImportError as e:
        raise ImportError(
            "Training requires ML dependencies. Install with: pip install harpocrates[ml]"
        ) from e

    # Prepare data
    X_train = np.array(train_data.features)
    y_train = np.array(train_data.labels)

    if val_data:
        X_val = np.array(val_data.features)
        y_val = np.array(val_data.labels)
    else:
        X_val, y_val = None, None

    # Handle class imbalance
    n_positive = sum(y_train)
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    if verbose:
        print(f"Training samples: {len(y_train)}")
        print(f"Class balance: {n_positive} positive, {n_negative} negative")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # XGBoost parameters optimized for:
    # - High precision (minimize false positives)
    # - Fast inference (shallow trees)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_child_weight": 5,  # Prevent overfitting
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    # Create and train model
    model = xgb.XGBClassifier(**params)

    if X_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose,
        )
    else:
        model.fit(X_train, y_train, verbose=verbose)

    # Find optimal threshold for target precision
    if X_val is not None:
        y_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

        # Find threshold that achieves target precision
        valid_idx = np.where(precision >= target_precision)[0]
        if len(valid_idx) > 0:
            # Take the threshold with highest recall among those meeting precision target
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            # If target precision not achievable, use threshold with best F1
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            if verbose:
                print(
                    f"Warning: Target precision {target_precision:.0%} not achievable. "
                    f"Using F1-optimal threshold."
                )
    else:
        optimal_threshold = 0.5

    # Compute final metrics
    if X_val is not None:
        y_pred = (y_proba >= optimal_threshold).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True)

        metrics = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc_roc": roc_auc_score(y_val, y_proba),
            "threshold": optimal_threshold,
        }

        if verbose:
            print("\nValidation Metrics:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.3f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print(f"  Optimal Threshold: {metrics['threshold']:.3f}")
    else:
        metrics = {"threshold": optimal_threshold}

    return model, optimal_threshold, metrics


def train_lightgbm_model(
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    target_precision: float = 0.95,
    verbose: bool = True,
) -> Tuple[Any, float, Dict[str, float]]:
    """
    Train LightGBM model.

    Args:
        train_data: Training dataset
        val_data: Optional validation dataset
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        n_estimators: Number of trees
        target_precision: Target precision for threshold tuning
        verbose: If True, print progress

    Returns:
        Tuple of (model, optimal_threshold, metrics)
    """
    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.metrics import (
            classification_report,
            precision_recall_curve,
            roc_auc_score,
        )
    except ImportError as e:
        raise ImportError(
            "Training requires ML dependencies. Install with: pip install harpocrates[ml]"
        ) from e

    # Prepare data
    X_train = np.array(train_data.features)
    y_train = np.array(train_data.labels)

    if val_data:
        X_val = np.array(val_data.features)
        y_val = np.array(val_data.labels)
    else:
        X_val, y_val = None, None

    # Handle class imbalance
    n_positive = sum(y_train)
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    if verbose:
        print(f"Training samples: {len(y_train)}")
        print(f"Class balance: {n_positive} positive, {n_negative} negative")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # LightGBM parameters
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_child_samples": 5,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1 if not verbose else 0,
    }

    # Create and train model
    model = lgb.LGBMClassifier(**params)

    if X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )
    else:
        model.fit(X_train, y_train)

    # Find optimal threshold for target precision
    if X_val is not None:
        y_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

        # Find threshold that achieves target precision
        valid_idx = np.where(precision >= target_precision)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            if verbose:
                print(
                    f"Warning: Target precision {target_precision:.0%} not achievable. "
                    f"Using F1-optimal threshold."
                )
    else:
        optimal_threshold = 0.5

    # Compute final metrics
    if X_val is not None:
        y_pred = (y_proba >= optimal_threshold).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True)

        metrics = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc_roc": roc_auc_score(y_val, y_proba),
            "threshold": optimal_threshold,
        }

        if verbose:
            print("\nValidation Metrics:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.3f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print(f"  Optimal Threshold: {metrics['threshold']:.3f}")
    else:
        metrics = {"threshold": optimal_threshold}

    return model, optimal_threshold, metrics


def train_ensemble(
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    target_precision: float = 0.95,
    verbose: bool = True,
) -> Tuple[Tuple[Any, Any], float, Dict[str, float]]:
    """
    Train ensemble of XGBoost and LightGBM models.

    Args:
        train_data: Training dataset
        val_data: Optional validation dataset
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        n_estimators: Number of trees
        target_precision: Target precision for threshold tuning
        verbose: If True, print progress

    Returns:
        Tuple of ((xgb_model, lgb_model), optimal_threshold, combined_metrics)
    """
    try:
        import numpy as np
        from sklearn.metrics import (
            classification_report,
            precision_recall_curve,
            roc_auc_score,
        )
    except ImportError as e:
        raise ImportError(
            "Training requires ML dependencies. Install with: pip install harpocrates[ml]"
        ) from e

    if verbose:
        print("=" * 50)
        print("Training XGBoost model...")
        print("=" * 50)

    xgb_model, xgb_threshold, xgb_metrics = train_model(
        train_data, val_data, max_depth, learning_rate, n_estimators,
        target_precision, verbose
    )

    if verbose:
        print("\n" + "=" * 50)
        print("Training LightGBM model...")
        print("=" * 50)

    lgb_model, lgb_threshold, lgb_metrics = train_lightgbm_model(
        train_data, val_data, max_depth, learning_rate, n_estimators,
        target_precision, verbose
    )

    # Combine predictions with weighted average
    if val_data:
        X_val = np.array(val_data.features)
        y_val = np.array(val_data.labels)

        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
        lgb_proba = lgb_model.predict_proba(X_val)[:, 1]

        # Ensemble: weighted average (XGBoost 0.6, LightGBM 0.4)
        ensemble_proba = 0.6 * xgb_proba + 0.4 * lgb_proba

        # Find optimal ensemble threshold
        precision, recall, thresholds = precision_recall_curve(y_val, ensemble_proba)
        valid_idx = np.where(precision >= target_precision)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred = (ensemble_proba >= optimal_threshold).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True)

        metrics = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc_roc": roc_auc_score(y_val, ensemble_proba),
            "threshold": optimal_threshold,
            "xgb_metrics": xgb_metrics,
            "lgb_metrics": lgb_metrics,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("Ensemble Metrics:")
            print("=" * 50)
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.3f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print(f"  Optimal Threshold: {metrics['threshold']:.3f}")
    else:
        optimal_threshold = 0.5
        metrics = {"threshold": optimal_threshold}

    return (xgb_model, lgb_model), optimal_threshold, metrics


def _to_python_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python_types(v) for v in obj]
    return obj


def save_model(
    model: Any,
    output_path: Path,
    threshold: float,
    metrics: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    model_type: str = "xgboost",
) -> None:
    """
    Save trained model and configuration.

    Args:
        model: Trained model (XGBoost, LightGBM, or tuple for ensemble)
        output_path: Path for model file
        threshold: Optimal classification threshold
        metrics: Training metrics
        feature_names: List of feature names
        model_type: Type of model ("xgboost", "lightgbm", or "ensemble")
    """
    # Convert numpy types to Python native types for JSON serialization
    threshold = _to_python_types(threshold)
    metrics = _to_python_types(metrics)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "ensemble":
        # Save both models for ensemble
        xgb_model, lgb_model = model
        xgb_path = output_path.parent / "xgboost_v1.json"
        lgb_path = output_path.parent / "lightgbm_v1.txt"

        xgb_model.save_model(str(xgb_path))
        lgb_model.booster_.save_model(str(lgb_path))

        config_path = output_path
        config = {
            "version": "1.0.0",
            "model_type": "ensemble",
            "threshold": threshold,
            "feature_names": feature_names or FeatureVector.get_feature_names(),
            "metrics": metrics,
            "xgboost_model": xgb_path.name,
            "lightgbm_model": lgb_path.name,
            "xgb_weight": 0.6,
            "lgb_weight": 0.4,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nXGBoost model saved to: {xgb_path}")
        print(f"LightGBM model saved to: {lgb_path}")
        print(f"Ensemble config saved to: {config_path}")

    elif model_type == "lightgbm":
        # Save LightGBM model
        model.booster_.save_model(str(output_path))

        config_path = output_path.with_name("lightgbm_config.json")
        config = {
            "version": "1.0.0",
            "model_type": "lightgbm",
            "threshold": threshold,
            "feature_names": feature_names or FeatureVector.get_feature_names(),
            "metrics": metrics,
            "model_file": output_path.name,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nModel saved to: {output_path}")
        print(f"Config saved to: {config_path}")

    else:
        # Save XGBoost model (default)
        model.save_model(str(output_path))

        config_path = output_path.with_name("feature_config.json")
        config = {
            "version": "1.0.0",
            "model_type": "xgboost",
            "threshold": threshold,
            "feature_names": feature_names or FeatureVector.get_feature_names(),
            "metrics": metrics,
            "model_file": output_path.name,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nModel saved to: {output_path}")
        print(f"Config saved to: {config_path}")


def get_feature_importance(model: Any) -> Dict[str, float]:
    """Get feature importance from trained model."""
    import numpy as np

    feature_names = FeatureVector.get_feature_names()
    importance = model.feature_importances_

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]

    return {feature_names[i]: float(importance[i]) for i in sorted_idx}


def print_feature_importance(model: Any, top_n: int = 10) -> None:
    """Print top N most important features."""
    importance = get_feature_importance(model)

    print(f"\nTop {top_n} Feature Importance:")
    for i, (name, score) in enumerate(list(importance.items())[:top_n], 1):
        print(f"  {i:2d}. {name}: {score:.4f}")


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train ML model for Harpocrates secrets detection"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        required=True,
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for model (auto-determined if not specified)",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Path to validation data (if not provided, will split training data)",
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        choices=MODEL_TYPES,
        default="xgboost",
        help="Model type: xgboost, lightgbm, or ensemble (default: xgboost)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees (default: 100)",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="Target precision for threshold tuning (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--cross-validate",
        "-cv",
        action="store_true",
        help="Use k-fold cross validation instead of single split",
    )
    parser.add_argument(
        "--folds",
        "-k",
        type=int,
        default=5,
        help="Number of folds for cross validation (default: 5)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Set default output path based on model type
    if args.output is None:
        # Use package directory to ensure scan command finds the models
        models_dir = Path(__file__).resolve().parent.parent / "ml" / "models"
        if args.model_type == "lightgbm":
            args.output = models_dir / "lightgbm_v1.txt"
        elif args.model_type == "ensemble":
            args.output = models_dir / "ensemble_config.json"
        else:
            args.output = models_dir / "xgboost_v1.json"

    print(f"Loading training data from {args.data}...")
    dataset = Dataset.from_jsonl(args.data)
    print(dataset.summary())

    # Use cross-validation if requested (only for XGBoost currently)
    if args.cross_validate:
        if args.model_type != "xgboost":
            print("Warning: Cross-validation currently only supports XGBoost. Using XGBoost.")
            args.model_type = "xgboost"

        from Harpocrates.training.cross_validation import cross_validate_with_best_model

        print(f"\nRunning {args.folds}-fold cross validation...")
        model, cv_result = cross_validate_with_best_model(
            dataset=dataset,
            k=args.folds,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            target_precision=args.target_precision,
            seed=args.seed,
            verbose=not args.quiet,
        )

        # Use mean threshold from CV
        threshold = cv_result.mean_threshold
        metrics = {
            "precision": cv_result.mean_precision,
            "recall": cv_result.mean_recall,
            "f1": cv_result.mean_f1,
            "auc_roc": cv_result.mean_auc_roc,
            "threshold": threshold,
            "cv_folds": args.folds,
            "cv_std_precision": cv_result.std_precision,
            "cv_std_recall": cv_result.std_recall,
        }
    else:
        # Load or create validation data
        if args.val_data:
            print(f"\nLoading validation data from {args.val_data}...")
            val_dataset = Dataset.from_jsonl(args.val_data)
            train_dataset = dataset
        else:
            print("\nSplitting into train/val sets (80/20)...")
            train_dataset, val_dataset, _ = dataset.split(
                train_ratio=0.8,
                val_ratio=0.2,
                seed=args.seed,
            )

        print(f"\nTrain: {len(train_dataset)} samples")
        print(f"Val: {len(val_dataset)} samples")

        # Train model based on type
        if args.model_type == "ensemble":
            print("\nTraining ensemble (XGBoost + LightGBM)...")
            model, threshold, metrics = train_ensemble(
                train_data=train_dataset,
                val_data=val_dataset,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                n_estimators=args.n_estimators,
                target_precision=args.target_precision,
                verbose=not args.quiet,
            )
        elif args.model_type == "lightgbm":
            print("\nTraining LightGBM model...")
            model, threshold, metrics = train_lightgbm_model(
                train_data=train_dataset,
                val_data=val_dataset,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                n_estimators=args.n_estimators,
                target_precision=args.target_precision,
                verbose=not args.quiet,
            )
        else:
            print("\nTraining XGBoost model...")
            model, threshold, metrics = train_model(
                train_data=train_dataset,
                val_data=val_dataset,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                n_estimators=args.n_estimators,
                target_precision=args.target_precision,
                verbose=not args.quiet,
            )

    # Print feature importance (only for single models)
    if not args.quiet and args.model_type != "ensemble":
        print_feature_importance(model)

    # Save model
    save_model(
        model=model,
        output_path=args.output,
        threshold=threshold,
        metrics=metrics,
        model_type=args.model_type,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
