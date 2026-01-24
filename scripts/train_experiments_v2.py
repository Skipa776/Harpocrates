#!/usr/bin/env python3
"""
Training experiments v2 - Focus on achieving 90% precision while maintaining 90% recall.

Current best: 85.26% precision, 95.32% recall
Gap: Need ~5% more precision without losing more than ~5% recall

Strategies:
1. More aggressive Stage B thresholds
2. Feature importance analysis and selection
3. Cost-sensitive learning with custom loss
4. Ensemble of multiple Stage B models
5. Post-processing calibration
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

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_data(
    data_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load training data from pickle file."""
    from Harpocrates.ml.features import FeatureVector, extract_features_from_record

    with open(data_path, "rb") as f:
        # WARNING: pickle can execute arbitrary code. Only load from trusted sources.
        records = pickle.load(f)  # nosec B301

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
    """Get indices of token-level features (first 23 features)."""
    return list(range(23))


def get_context_feature_indices() -> List[int]:
    """Get indices of context features (features 33-50, after token and variable-name)."""
    return list(range(33, 51))


def analyze_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Analyze feature importance and correlation with errors."""
    import lightgbm as lgb
    from sklearn.metrics import precision_score, recall_score
    from Harpocrates.ml.features import FeatureVector

    feature_names = FeatureVector.get_feature_names()

    # Train a quick model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Get predictions and errors
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # False positives and false negatives
    fp_mask = (y_pred == 1) & (y_val == 0)
    fn_mask = (y_pred == 0) & (y_val == 1)

    # Feature importance
    importance = model.feature_importances_
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}

    # Sort by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Analyze features correlated with FPs
    fp_feature_means = X_val[fp_mask].mean(axis=0)
    tp_feature_means = X_val[(y_pred == 1) & (y_val == 1)].mean(axis=0)

    # Features where FP mean differs significantly from TP mean
    fp_indicators = {}
    for i, name in enumerate(feature_names):
        if fp_mask.sum() > 0 and ((y_pred == 1) & (y_val == 1)).sum() > 0:
            diff = fp_feature_means[i] - tp_feature_means[i]
            fp_indicators[name] = float(diff)

    return {
        "importance": sorted_importance[:15],
        "fp_count": int(fp_mask.sum()),
        "fn_count": int(fn_mask.sum()),
        "fp_indicators": sorted(fp_indicators.items(), key=lambda x: abs(x[1]), reverse=True)[:10],
    }


def train_with_focal_loss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> Tuple[Any, Dict[str, float]]:
    """Train with focal loss for harder examples."""
    import lightgbm as lgb
    from sklearn.metrics import precision_score, recall_score, f1_score

    # LightGBM with focal loss via custom objective
    def focal_loss_lgb(y_true, y_pred):
        """Focal loss for LightGBM with correct per-class derivatives."""
        eps = 1e-10
        p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
        p = np.clip(p, eps, 1.0 - eps)

        # Positive samples (y=1): -alpha * (1-p)^gamma * log(p)
        # Negative samples (y=0): -(1-alpha) * p^gamma * log(1-p)
        grad = np.where(
            y_true == 1,
            alpha * (1 - p) ** gamma * (gamma * p * np.log(p) + p - 1),
            (1 - alpha) * p ** gamma * (-gamma * (1 - p) * np.log(1 - p) + p),
        )
        hess = np.where(
            y_true == 1,
            alpha * (1 - p) ** gamma * p * (
                (1 - p) - gamma * (1 - p) * np.log(p) * (gamma * p - 1 + p)
                + gamma * p * (1 - p)
            ),
            (1 - alpha) * p ** gamma * (1 - p) * (
                p + gamma * p * np.log(1 - p) * (gamma * (1 - p) - p)
                + gamma * (1 - p) * p
            ),
        )
        # Ensure hessians are positive for numerical stability
        hess = np.maximum(hess, eps)
        return grad, hess

    model = lgb.LGBMClassifier(
        objective=focal_loss_lgb,
        n_estimators=200,
        max_depth=10,
        num_leaves=47,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]

    # Find optimal threshold for 90% recall minimum
    best_precision = 0
    best_threshold = 0.5
    for thresh in np.arange(0.3, 0.8, 0.02):
        y_pred = (y_proba > thresh).astype(int)
        if sum(y_pred) == 0:
            continue
        r = recall_score(y_val, y_pred)
        if r >= 0.90:
            p = precision_score(y_val, y_pred)
            if p > best_precision:
                best_precision = p
                best_threshold = thresh

    y_pred = (y_proba > best_threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "threshold": float(best_threshold),
    }

    return model, metrics


def train_ensemble_stage_b(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[List[Any], Dict[str, float]]:
    """Train ensemble of Stage B models."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score

    models = []

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        num_leaves=47,
        learning_rate=0.05,
        scale_pos_weight=0.7,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    models.append(("lightgbm", lgb_model))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=0.7,
        random_state=43,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    models.append(("xgboost", xgb_model))

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight={0: 1.0, 1: 0.7},
        random_state=44,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    models.append(("random_forest", rf_model))

    # Ensemble predictions (average)
    probas = []
    for name, model in models:
        probas.append(model.predict_proba(X_val)[:, 1])
    ensemble_proba = np.mean(probas, axis=0)

    # Find optimal threshold
    best_precision = 0
    best_threshold = 0.5
    for thresh in np.arange(0.3, 0.8, 0.02):
        y_pred = (ensemble_proba > thresh).astype(int)
        if sum(y_pred) == 0:
            continue
        r = recall_score(y_val, y_pred)
        if r >= 0.90:
            p = precision_score(y_val, y_pred)
            if p > best_precision:
                best_precision = p
                best_threshold = thresh

    y_pred = (ensemble_proba > best_threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "threshold": float(best_threshold),
    }

    return models, metrics


def train_with_high_precision_focus(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    stage_a_model: Any,
    stage_a_threshold_low: float = 0.15,
    stage_a_threshold_high: float = 0.85,
) -> Dict[str, Any]:
    """Train with focus on high precision while maintaining 90% recall."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    token_indices = get_token_feature_indices()
    X_train_tokens = X_train[:, token_indices]
    X_val_tokens = X_val[:, token_indices]

    # Get Stage A probabilities
    stage_a_train_proba = stage_a_model.predict_proba(X_train_tokens)[:, 1]
    stage_a_val_proba = stage_a_model.predict_proba(X_val_tokens)[:, 1]

    # Filter training to ambiguous samples
    ambiguous_mask_train = (stage_a_train_proba >= stage_a_threshold_low) & (
        stage_a_train_proba <= stage_a_threshold_high
    )
    X_train_amb = X_train[ambiguous_mask_train]
    y_train_amb = y_train[ambiguous_mask_train]

    # Guard: empty or single-class ambiguous subsets make Stage B training invalid
    if len(y_train_amb) == 0 or len(np.unique(y_train_amb)) < 2:
        return None

    # Train Stage B with very conservative settings
    n_pos = sum(y_train_amb)
    n_neg = len(y_train_amb) - n_pos
    scale_weight = (n_neg / n_pos) * 0.5 if n_pos > 0 else 1.0  # Very conservative

    stage_b_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=12,
        num_leaves=63,
        learning_rate=0.03,
        min_child_samples=30,
        scale_pos_weight=scale_weight,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=42,
        verbose=-1,
    )
    stage_b_model.fit(X_train_amb, y_train_amb)

    stage_b_val_proba = stage_b_model.predict_proba(X_val)[:, 1]

    # Search for best threshold combination
    best_result = None
    best_f1 = 0

    for stage_b_thresh in np.arange(0.3, 0.8, 0.02):
        # Combined predictions
        y_pred = np.zeros(len(y_val), dtype=int)

        low_mask = stage_a_val_proba < stage_a_threshold_low
        high_mask = stage_a_val_proba > stage_a_threshold_high
        ambiguous_mask = ~low_mask & ~high_mask

        y_pred[low_mask] = 0
        y_pred[high_mask] = 1
        y_pred[ambiguous_mask] = (stage_b_val_proba[ambiguous_mask] > stage_b_thresh).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        # We want precision >= 90% and recall >= 90%
        if precision >= 0.90 and recall >= 0.90:
            if f1 > best_f1:
                best_f1 = f1
                best_result = {
                    "stage_b_threshold": float(stage_b_thresh),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "model": stage_b_model,
                    "targets_met": True,
                }
        elif f1 > best_f1:
            best_f1 = f1
            best_result = {
                "stage_b_threshold": float(stage_b_thresh),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "model": stage_b_model,
                "targets_met": precision >= 0.90 and recall >= 0.90,
            }

    return best_result


def run_v2_experiments():
    """Run v2 experiments focused on precision."""
    print("=" * 60)
    print("HARPOCRATES ML EXPERIMENTS V2 - PRECISION FOCUS")
    print("=" * 60)
    print("Target: Precision >= 90%, Recall >= 90%")
    print("Current best: 85.26% precision, 95.32% recall")
    print()

    # Load data
    data_path = Path("Harpocrates/training/data/training_data_v2.pkl")
    print("Loading training data...")
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

    # 1. Analyze feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    importance_analysis = analyze_feature_importance(X_train, y_train, X_val, y_val)

    print("\nTop 15 most important features:")
    for name, imp in importance_analysis["importance"]:
        print(f"  {name}: {imp:.4f}")

    print(f"\nFalse positives: {importance_analysis['fp_count']}")
    print(f"False negatives: {importance_analysis['fn_count']}")

    print("\nFeatures most correlated with false positives:")
    for name, diff in importance_analysis["fp_indicators"]:
        direction = "higher" if diff > 0 else "lower"
        print(f"  {name}: {direction} in FPs by {abs(diff):.4f}")

    # 2. Train Stage A
    print("\n" + "=" * 60)
    print("TRAINING STAGE A (HIGH RECALL)")
    print("=" * 60)

    import xgboost as xgb
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    token_indices = get_token_feature_indices()
    X_train_tokens = X_train[:, token_indices]
    X_val_tokens = X_val[:, token_indices]

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos

    if n_pos == 0:
        raise ValueError("No positive samples in training data; cannot train Stage A")

    stage_a_model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=5,
        learning_rate=0.1,
        n_estimators=120,
        min_child_weight=3,
        scale_pos_weight=(n_neg / n_pos) * 1.8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    stage_a_model.fit(X_train_tokens, y_train, verbose=False)

    stage_a_proba = stage_a_model.predict_proba(X_val_tokens)[:, 1]
    print(f"Stage A AUC: {roc_auc_score(y_val, stage_a_proba):.4f}")

    # 3. Test multiple Stage A threshold combinations
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)

    results = []

    threshold_combos = [
        (0.10, 0.90),
        (0.12, 0.88),
        (0.15, 0.85),
        (0.18, 0.82),
        (0.20, 0.80),
        (0.08, 0.92),
        (0.05, 0.95),
    ]

    for low, high in threshold_combos:
        result = train_with_high_precision_focus(
            X_train, y_train, X_val, y_val,
            stage_a_model, low, high
        )
        if result:
            result["stage_a_low"] = low
            result["stage_a_high"] = high
            results.append(result)
            print(f"Thresholds ({low:.2f}, {high:.2f}): "
                  f"P={result['precision']:.2%}, R={result['recall']:.2%}, "
                  f"F1={result['f1']:.4f} {'***' if result['targets_met'] else ''}")

    # 4. Try ensemble approach
    print("\n" + "=" * 60)
    print("ENSEMBLE STAGE B")
    print("=" * 60)

    ensemble_models, ensemble_metrics = train_ensemble_stage_b(X_train, y_train, X_val, y_val)
    print(f"Ensemble: P={ensemble_metrics['precision']:.2%}, "
          f"R={ensemble_metrics['recall']:.2%}, F1={ensemble_metrics['f1']:.4f}")

    # Find best overall result
    print("\n" + "=" * 60)
    print("BEST RESULTS")
    print("=" * 60)

    all_results = sorted(results, key=lambda x: x["f1"], reverse=True)

    if not all_results:
        print("\nNo valid results produced. Check Stage A/B configurations.")
        return

    # Check for any that meet targets
    target_met = [r for r in all_results if r["targets_met"]]
    if target_met:
        print("\n*** CONFIGURATIONS MEETING TARGETS ***")
        for r in target_met:
            print(f"  Stage A: ({r['stage_a_low']:.2f}, {r['stage_a_high']:.2f})")
            print(f"  Stage B threshold: {r['stage_b_threshold']:.2f}")
            print(f"  Precision: {r['precision']:.2%}")
            print(f"  Recall: {r['recall']:.2%}")
            print(f"  F1: {r['f1']:.4f}")
            print()
    else:
        print("\nNo configuration met both targets yet.")
        print("\nBest by F1:")
        best = all_results[0]
        print(f"  Stage A: ({best['stage_a_low']:.2f}, {best['stage_a_high']:.2f})")
        print(f"  Stage B threshold: {best['stage_b_threshold']:.2f}")
        print(f"  Precision: {best['precision']:.2%}")
        print(f"  Recall: {best['recall']:.2%}")
        print(f"  F1: {best['f1']:.4f}")

        # Best by precision while maintaining recall >= 90%
        high_recall = [r for r in all_results if r["recall"] >= 0.90]
        if high_recall:
            best_prec = max(high_recall, key=lambda x: x["precision"])
            print(f"\nBest precision with recall >= 90%:")
            print(f"  Stage A: ({best_prec['stage_a_low']:.2f}, {best_prec['stage_a_high']:.2f})")
            print(f"  Stage B threshold: {best_prec['stage_b_threshold']:.2f}")
            print(f"  Precision: {best_prec['precision']:.2%}")
            print(f"  Recall: {best_prec['recall']:.2%}")
            print(f"  F1: {best_prec['f1']:.4f}")

    # Save best model
    if all_results:
        best = all_results[0]
        output_dir = Path("Harpocrates/ml/models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Stage A
        stage_a_path = output_dir / "stageA_xgboost.json"
        stage_a_model.save_model(str(stage_a_path))

        # Save Stage B
        stage_b_path = output_dir / "stageB_lightgbm.txt"
        best["model"].booster_.save_model(str(stage_b_path))

        # Save config
        config = {
            "version": "2.2.0",
            "mode": "two_stage",
            "created_at": datetime.now().isoformat(),
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
                "feature_count": X_train.shape[1],
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

        print(f"\nSaved models to {output_dir}")


if __name__ == "__main__":
    run_v2_experiments()
