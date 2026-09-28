"""
K-Fold Cross Validation for Harpocrates ML training.

Provides stratified k-fold cross validation to evaluate model performance
with reliable metrics that aren't biased by a single train/test split.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from Harpocrates.training.dataset import Dataset


@dataclass
class FoldMetrics:
    """Metrics from a single fold evaluation."""

    fold: int
    precision: float
    recall: float
    f1: float
    auc_roc: float
    threshold: float
    train_size: int
    val_size: int


@dataclass
class CrossValidationResult:
    """Aggregated results from k-fold cross validation."""

    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    @property
    def mean_precision(self) -> float:
        """Mean precision across folds."""
        return sum(f.precision for f in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_recall(self) -> float:
        """Mean recall across folds."""
        return sum(f.recall for f in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_f1(self) -> float:
        """Mean F1 score across folds."""
        return sum(f.f1 for f in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_auc_roc(self) -> float:
        """Mean AUC-ROC across folds."""
        return sum(f.auc_roc for f in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_threshold(self) -> float:
        """Mean optimal threshold across folds."""
        return sum(f.threshold for f in self.fold_metrics) / len(self.fold_metrics)

    @property
    def std_precision(self) -> float:
        """Standard deviation of precision across folds."""
        mean = self.mean_precision
        return (sum((f.precision - mean) ** 2 for f in self.fold_metrics) / len(self.fold_metrics)) ** 0.5

    @property
    def std_recall(self) -> float:
        """Standard deviation of recall across folds."""
        mean = self.mean_recall
        return (sum((f.recall - mean) ** 2 for f in self.fold_metrics) / len(self.fold_metrics)) ** 0.5

    @property
    def std_f1(self) -> float:
        """Standard deviation of F1 across folds."""
        mean = self.mean_f1
        return (sum((f.f1 - mean) ** 2 for f in self.fold_metrics) / len(self.fold_metrics)) ** 0.5

    def summary(self) -> str:
        """Get summary string of CV results."""
        lines = [
            f"Cross Validation Results ({len(self.fold_metrics)} folds):",
            f"  Precision: {self.mean_precision:.3f} (+/- {self.std_precision:.3f})",
            f"  Recall:    {self.mean_recall:.3f} (+/- {self.std_recall:.3f})",
            f"  F1 Score:  {self.mean_f1:.3f} (+/- {self.std_f1:.3f})",
            f"  AUC-ROC:   {self.mean_auc_roc:.3f}",
            f"  Threshold: {self.mean_threshold:.3f}",
            "",
            "Per-fold metrics:",
        ]
        for fm in self.fold_metrics:
            lines.append(
                f"  Fold {fm.fold}: P={fm.precision:.3f} R={fm.recall:.3f} "
                f"F1={fm.f1:.3f} (train={fm.train_size}, val={fm.val_size})"
            )
        return "\n".join(lines)


def stratified_k_fold_split(
    records: List[Dict[str, Any]],
    k: int = 5,
    seed: Optional[int] = None,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Create stratified k-fold splits.

    Maintains class balance in each fold by separately splitting
    positive and negative samples.

    Args:
        records: List of training records
        k: Number of folds
        seed: Random seed for reproducibility

    Returns:
        List of (train_records, val_records) tuples, one per fold
    """
    if seed is not None:
        random.seed(seed)

    # Separate by label
    positive = [r for r in records if r["label"] == 1]
    negative = [r for r in records if r["label"] == 0]

    # Shuffle each class
    random.shuffle(positive)
    random.shuffle(negative)

    # Create fold indices for each class
    def create_fold_indices(n: int, k: int) -> List[List[int]]:
        """Create k roughly equal-sized groups of indices."""
        fold_size = n // k
        remainder = n % k

        folds = []
        start = 0
        for i in range(k):
            # Distribute remainder among first 'remainder' folds
            end = start + fold_size + (1 if i < remainder else 0)
            folds.append(list(range(start, end)))
            start = end
        return folds

    pos_folds = create_fold_indices(len(positive), k)
    neg_folds = create_fold_indices(len(negative), k)

    # Create train/val splits for each fold
    splits = []
    for fold_idx in range(k):
        # Validation set = samples in this fold
        val_pos = [positive[i] for i in pos_folds[fold_idx]]
        val_neg = [negative[i] for i in neg_folds[fold_idx]]

        # Training set = samples in all other folds
        train_pos = []
        train_neg = []
        for other_fold in range(k):
            if other_fold != fold_idx:
                train_pos.extend([positive[i] for i in pos_folds[other_fold]])
                train_neg.extend([negative[i] for i in neg_folds[other_fold]])

        # Combine and shuffle
        train = train_pos + train_neg
        val = val_pos + val_neg
        random.shuffle(train)
        random.shuffle(val)

        splits.append((train, val))

    return splits


def _train_xgboost(
    X_train, y_train, X_val, y_val,
    max_depth: int, learning_rate: float, n_estimators: int,
    scale_pos_weight: float, seed: int, verbose: bool = False,
):
    """Train an XGBoost model and return it with predictions."""
    import xgboost as xgb

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_child_weight": 5,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    y_proba = model.predict_proba(X_val)[:, 1]
    return model, y_proba


def _train_lightgbm(
    X_train, y_train, X_val, y_val,
    max_depth: int, learning_rate: float, n_estimators: int,
    scale_pos_weight: float, seed: int, verbose: bool = False,
):
    """Train a LightGBM model and return it with predictions."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_proba = model.predict_proba(X_val)[:, 1]
    return model, y_proba


def _train_ensemble(
    X_train, y_train, X_val, y_val,
    max_depth: int, learning_rate: float, n_estimators: int,
    scale_pos_weight: float, seed: int, verbose: bool = False,
    xgb_weight: float = 0.6, lgb_weight: float = 0.4,
):
    """Train an ensemble of XGBoost and LightGBM, return combined predictions."""
    xgb_model, xgb_proba = _train_xgboost(
        X_train, y_train, X_val, y_val,
        max_depth, learning_rate, n_estimators,
        scale_pos_weight, seed, verbose
    )
    lgb_model, lgb_proba = _train_lightgbm(
        X_train, y_train, X_val, y_val,
        max_depth, learning_rate, n_estimators,
        scale_pos_weight, seed, verbose
    )
    # Weighted average
    y_proba = xgb_weight * xgb_proba + lgb_weight * lgb_proba
    return (xgb_model, lgb_model), y_proba


def cross_validate(
    dataset: Dataset,
    k: int = 5,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    target_precision: float = 0.95,
    seed: Optional[int] = None,
    verbose: bool = True,
    model_type: str = "xgboost",
) -> CrossValidationResult:
    """
    Perform k-fold cross validation.

    Args:
        dataset: Dataset to evaluate
        k: Number of folds
        max_depth: Max tree depth
        learning_rate: Learning rate
        n_estimators: Number of trees
        target_precision: Target precision for threshold tuning
        seed: Random seed for reproducibility
        verbose: If True, print progress
        model_type: Model type - "xgboost", "lightgbm", or "ensemble"

    Returns:
        CrossValidationResult with per-fold and aggregated metrics
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
            "Cross validation requires ML dependencies. "
            "Install with: pip install harpocrates[ml]"
        ) from e

    # Create stratified splits
    splits = stratified_k_fold_split(dataset.records, k=k, seed=seed)

    result = CrossValidationResult()

    for fold_idx, (train_records, val_records) in enumerate(splits):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{k} ---")

        # Create datasets for this fold
        train_data = Dataset(train_records, name=f"fold{fold_idx}_train")
        val_data = Dataset(val_records, name=f"fold{fold_idx}_val")

        # Prepare data
        X_train = np.array(train_data.features)
        y_train = np.array(train_data.labels)
        X_val = np.array(val_data.features)
        y_val = np.array(val_data.labels)

        # Handle class imbalance
        n_positive = sum(y_train)
        n_negative = len(y_train) - n_positive
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        if verbose:
            print(f"Train: {len(y_train)} samples ({n_positive} pos, {n_negative} neg)")
            print(f"Val: {len(y_val)} samples")

        # Train model based on type
        effective_seed = seed if seed else 42
        if model_type == "lightgbm":
            model, y_proba = _train_lightgbm(
                X_train, y_train, X_val, y_val,
                max_depth, learning_rate, n_estimators,
                scale_pos_weight, effective_seed, False
            )
        elif model_type == "ensemble":
            model, y_proba = _train_ensemble(
                X_train, y_train, X_val, y_val,
                max_depth, learning_rate, n_estimators,
                scale_pos_weight, effective_seed, False
            )
        else:  # xgboost
            model, y_proba = _train_xgboost(
                X_train, y_train, X_val, y_val,
                max_depth, learning_rate, n_estimators,
                scale_pos_weight, effective_seed, False
            )

        # Find optimal threshold for target precision
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

        valid_idx = np.where(precision >= target_precision)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        # Compute metrics at optimal threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        fold_metrics = FoldMetrics(
            fold=fold_idx + 1,
            precision=report["1"]["precision"],
            recall=report["1"]["recall"],
            f1=report["1"]["f1-score"],
            auc_roc=roc_auc_score(y_val, y_proba),
            threshold=optimal_threshold,
            train_size=len(train_records),
            val_size=len(val_records),
        )

        result.fold_metrics.append(fold_metrics)

        if verbose:
            print(f"Precision: {fold_metrics.precision:.3f}")
            print(f"Recall: {fold_metrics.recall:.3f}")
            print(f"F1: {fold_metrics.f1:.3f}")

    if verbose:
        print("\n" + result.summary())

    return result


def cross_validate_with_best_model(
    dataset: Dataset,
    k: int = 5,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    target_precision: float = 0.95,
    seed: Optional[int] = None,
    verbose: bool = True,
    model_type: str = "xgboost",
) -> Tuple[Any, CrossValidationResult]:
    """
    Perform k-fold CV and return the best performing model.

    After cross validation, retrains on the full dataset using
    the average optimal threshold from CV.

    Args:
        dataset: Dataset to evaluate
        k: Number of folds
        max_depth: Max tree depth
        learning_rate: Learning rate
        n_estimators: Number of trees
        target_precision: Target precision for threshold tuning
        seed: Random seed for reproducibility
        verbose: If True, print progress
        model_type: Model type - "xgboost", "lightgbm", or "ensemble"

    Returns:
        Tuple of (best_model, cv_results)
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "Cross validation requires ML dependencies. "
            "Install with: pip install harpocrates[ml]"
        ) from e

    # Run cross validation
    cv_result = cross_validate(
        dataset=dataset,
        k=k,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        target_precision=target_precision,
        seed=seed,
        verbose=verbose,
        model_type=model_type,
    )

    if verbose:
        print("\nRetraining on full dataset...")

    # Train final model on full dataset
    X = np.array(dataset.features)
    y = np.array(dataset.labels)

    n_positive = sum(y)
    n_negative = len(y) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    effective_seed = seed if seed else 42

    # Train based on model type
    if model_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="binary",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=effective_seed,
            verbose=-1,
        )
        model.fit(X, y)
    elif model_type == "ensemble":
        import lightgbm as lgb
        import xgboost as xgb

        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "aucpr"],
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=effective_seed,
        )
        xgb_model.fit(X, y, verbose=False)

        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=effective_seed,
            verbose=-1,
        )
        lgb_model.fit(X, y)

        model = (xgb_model, lgb_model)
    else:  # xgboost
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "aucpr"],
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=effective_seed,
        )
        model.fit(X, y, verbose=False)

    if verbose:
        print(f"Final model trained on {len(y)} samples")
        print(f"Using threshold from CV: {cv_result.mean_threshold:.3f}")

    return model, cv_result


__all__ = [
    "FoldMetrics",
    "CrossValidationResult",
    "stratified_k_fold_split",
    "cross_validate",
    "cross_validate_with_best_model",
]
