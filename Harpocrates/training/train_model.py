"""
Single-stage XGBoost training for Harpocrates ML.

Trains a single XGBoost model on all 65 features with Platt scaling
for calibrated probabilities and dual-threshold decision logic.

Usage (auto-split 65/10/10/15, with OOD golden eval):
    python -m Harpocrates.training.train_model \\
        --data data/synthetic_v2_full.jsonl \\
        --golden-data data/golden_test.jsonl \\
        --output-dir Harpocrates/ml/models \\
        --seed 42

Usage (explicit splits):
    python -m Harpocrates.training.train_model \\
        --train-data train.jsonl \\
        --val-data val.jsonl \\
        --cal-data cal.jsonl \\
        --test-data test.jsonl \\
        --output-dir Harpocrates/ml/models \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

N_FEATURES = 65


# ---------------------------------------------------------------------------
# Phase 1 helpers: load records (no feature extraction yet)
# ---------------------------------------------------------------------------


def load_records(
    path: Path, deduplicate: bool = True
) -> Tuple[List[dict], Set[str], Set[str]]:
    """Load JSONL and optionally deduplicate within the file.

    Returns (records, record_hashes, tokens) where:
    - record_hashes: sha1(token + "|" + line_content) per surviving record
    - tokens: unique token strings from surviving records
    """
    records: List[dict] = []
    record_hashes: Set[str] = set()
    tokens: Set[str] = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            h = sha1(
                f"{record['token']}|{record['line_content']}".encode()
            ).hexdigest()
            if deduplicate and h in record_hashes:
                continue
            record_hashes.add(h)
            tokens.add(record["token"])
            records.append(record)

    return records, record_hashes, tokens


def extract_features_from_records(
    records: List[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 65-dim feature vectors and labels from in-memory records."""
    from Harpocrates.ml.features import extract_features_from_record

    features: List[List[float]] = []
    labels: List[int] = []

    for i, record in enumerate(records):
        try:
            fv = extract_features_from_record(record)
            arr = fv.to_array()
            if len(arr) != N_FEATURES:
                print(
                    f"Warning: Expected {N_FEATURES} features, got {len(arr)} "
                    f"(record {i}). Skipping."
                )
                continue
            features.append(arr)
            labels.append(record["label"])
        except Exception as e:
            print(f"Warning: Feature extraction failed for record {i}: {e}")
            continue

    return np.array(features), np.array(labels)


# ---------------------------------------------------------------------------
# Phase 2 helper: cross-split overlap protection
# ---------------------------------------------------------------------------


def check_cross_split_overlap(
    split_records: Dict[str, List[dict]],
    split_record_hashes: Dict[str, Set[str]],
    split_tokens: Dict[str, Set[str]],
    max_drop_pct: float = 0.20,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """Three-tier cross-split protection.

    Exact record-hash overlap between any two splits → raises ValueError.
    Token-only overlap → records dropped from the lower-priority split
    (train stays pristine; val > cal > test in priority order).
    Drop budget: if token-dropping would remove > max_drop_pct of any
    held-out split → raises ValueError instead of silently shrinking.
    """
    priority = {"train": 0, "val": 1, "cal": 2, "test": 3}
    splits = sorted(split_records.keys(), key=lambda s: priority.get(s, 99))

    filtered_records = {k: list(v) for k, v in split_records.items()}
    filtered_tokens: Dict[str, Set[str]] = {k: set(v) for k, v in split_tokens.items()}

    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1 :]:
            pa = priority.get(split_a, 99)
            pb = priority.get(split_b, 99)
            higher, lower = (split_a, split_b) if pa <= pb else (split_b, split_a)

            # Exact duplicate check — always fatal.
            hash_overlap = split_record_hashes[split_a] & split_record_hashes[split_b]
            if hash_overlap:
                samples = list(hash_overlap)[:5]
                raise ValueError(
                    f"Exact duplicate records between '{split_a}' and '{split_b}': "
                    f"{len(hash_overlap)} duplicates. First 5 hashes: {samples}"
                )

            # Token-only overlap — drop from the lower-priority split.
            token_overlap = filtered_tokens[higher] & filtered_tokens[lower]
            if not token_overlap:
                continue

            current = filtered_records[lower]
            would_drop = sum(1 for r in current if r["token"] in token_overlap)
            if current:
                drop_pct = would_drop / len(current)
                if drop_pct > max_drop_pct:
                    samples = list(token_overlap)[:5]
                    raise ValueError(
                        f"Token overlap between '{higher}' and '{lower}' would drop "
                        f"{drop_pct:.1%} ({would_drop}/{len(current)} records) — "
                        f"exceeds max_drop_pct={max_drop_pct:.0%}. "
                        f"Data generator is producing too much overlap. "
                        f"First 5 overlapping tokens: {samples}"
                    )

            filtered_records[lower] = [
                r for r in current if r["token"] not in token_overlap
            ]
            if verbose:
                print(
                    f"  Cross-split: dropped {would_drop} token-overlapping records "
                    f"from '{lower}' (overlap with '{higher}')"
                )
            filtered_tokens[lower] = {r["token"] for r in filtered_records[lower]}

    return filtered_records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_feature_importance(
    model: Any,
    top_n: int = 10,
    dominance_threshold: float = 0.80,
    verbose: bool = True,
) -> Dict[str, float]:
    """Print top feature importances and warn if a single feature dominates."""
    from Harpocrates.ml.features import FeatureVector

    importances = model.feature_importances_
    feature_names = FeatureVector.get_feature_names()
    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    top = pairs[:top_n]

    if verbose:
        print(f"\n=== FEATURE IMPORTANCE (Top {top_n}) ===")
        for name, imp in top:
            print(f"  {name:<42} {imp:.4f}")

    total = float(importances.sum())
    if total > 0:
        top1_pct = float(importances.max()) / total
        if top1_pct > dominance_threshold:
            top1_name = feature_names[int(importances.argmax())]
            print(
                f"\nWARNING: '{top1_name}' accounts for {top1_pct:.1%} of total "
                f"importance — possible leakage signal."
            )

    return {name: float(imp) for name, imp in top}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
            print(f"\nRaw XGBoost AUC-ROC (val): {metrics['auc_roc']:.4f}")

    return model, metrics


def calibrate_model(
    model: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    verbose: bool = True,
) -> Any:
    """Fit Platt scaling on the calibration set for calibrated probabilities.

    The calibration set must be disjoint from the validation set used for
    early stopping. After calibration, P=0.85 means "85% of tokens with
    this score are actually secrets."
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_cal, y_cal)

    if verbose:
        raw_proba = model.predict_proba(X_cal)[:, 1]
        cal_proba = calibrator.predict_proba(X_cal)[:, 1]
        raw_brier = brier_score_loss(y_cal, raw_proba)
        cal_brier = brier_score_loss(y_cal, cal_proba)
        print("\n=== PLATT SCALING ===")
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
    f1_scores = (
        2
        * precisions[:-1]
        * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])

    if verbose:
        print("\n=== DUAL THRESHOLDS ===")
        print(
            f"Threshold low  (recall >= {target_recall:.0%}):    {threshold_low:.4f}"
        )
        print(
            f"Threshold high (precision >= {target_precision:.0%}): {threshold_high:.4f}"
        )
        print(f"Optimal F1 threshold:                    {optimal_threshold:.4f}")

        review_mask = (y_proba >= threshold_low) & (y_proba <= threshold_high)
        review_pct = review_mask.sum() / len(y_proba) * 100
        print(f"Review zone:  {review_mask.sum()}/{len(y_proba)} ({review_pct:.1f}%)")

    return threshold_low, threshold_high, optimal_threshold


def evaluate(
    calibrator: Any,
    threshold_low: float,
    threshold_high: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str = "TEST",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate model with dual-threshold decision logic on a held-out test set.

    `label` is used only in printed headers to distinguish evaluation passes
    (e.g. "SYNTHETIC TEST" vs "GOLDEN OOD").
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_proba = calibrator.predict_proba(X_test)[:, 1]

    safe_mask = y_proba < threshold_low
    secret_mask = y_proba > threshold_high
    review_mask = ~safe_mask & ~secret_mask

    # REVIEW and SECRET both count as "flagged"
    y_pred = np.where(safe_mask, 0, 1)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)

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
        "review_zone_pct": float(review_mask.sum() / len(y_test) * 100),
    }

    if verbose:
        print(f"\n=== EVALUATION: {label} ===")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"AUC-ROC:   {auc_roc:.4f}")
        print("\nRouting breakdown:")
        print(f"  SAFE:    {safe_mask.sum()} ({safe_mask.sum()/len(y_test):.1%})")
        print(f"  REVIEW:  {review_mask.sum()} ({review_mask.sum()/len(y_test):.1%})")
        print(f"  SECRET:  {secret_mask.sum()} ({secret_mask.sum()/len(y_test):.1%})")
        print(
            f"\n{classification_report(y_test, y_pred, target_names=['Non-Secret', 'Secret'])}"
        )
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
        print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    return metrics


# ---------------------------------------------------------------------------
# Serialization + persistence
# ---------------------------------------------------------------------------


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

    model_path = output_dir / "stageA_xgboost.json"
    model.save_model(str(model_path))

    config_path = output_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_convert_to_serializable(config), f, indent=2)

    if verbose:
        print("\nModel saved:")
        print(f"  XGBoost: {model_path}")
        print(f"  Config:  {config_path}")

    return {"model": model_path, "config": config_path}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train single-stage XGBoost model for secret detection"
    )

    # Data source: --data (auto-split) OR --train-data (explicit splits).
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--data",
        "-d",
        type=Path,
        metavar="JSONL",
        help="Single JSONL; auto-split 65/10/10/15 (train/val/cal/test)",
    )
    source_group.add_argument(
        "--train-data",
        "-t",
        type=Path,
        metavar="JSONL",
        help="Training data JSONL",
    )
    parser.add_argument(
        "--val-data",
        "-v",
        type=Path,
        metavar="JSONL",
        help="Validation JSONL (early stopping only)",
    )
    parser.add_argument(
        "--cal-data",
        type=Path,
        metavar="JSONL",
        help="Calibration JSONL (Platt scaling + threshold search only)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        metavar="JSONL",
        help="Test JSONL (final synthetic evaluation only)",
    )
    parser.add_argument(
        "--golden-data",
        type=Path,
        default=None,
        metavar="JSONL",
        help="OOD golden test set JSONL — quarantined, evaluated independently",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("Harpocrates/ml/models"),
        help="Output directory for models",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        "--max-overlap",
        type=float,
        default=0.20,
        metavar="FRAC",
        help="Max fraction of a held-out split that may be dropped due to token overlap "
        "before raising an error (default: 0.20)",
    )
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    try:
        import xgboost  # noqa: F401
    except ImportError as e:
        print("Error: Missing ML dependency. Install with: pip install harpocrates[ml]")
        print(f"Details: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Phase 1: Load records (no feature extraction yet).                  #
    # ------------------------------------------------------------------ #
    split_records: Dict[str, List[dict]] = {}
    split_hashes: Dict[str, Set[str]] = {}
    split_tokens: Dict[str, Set[str]] = {}

    if args.data:
        from Harpocrates.training.dataset import train_val_cal_test_split

        if verbose:
            print(f"Loading {args.data} for auto 65/10/10/15 split …")
        all_records, _, _ = load_records(args.data, deduplicate=True)
        if verbose:
            print(f"  {len(all_records)} records after intra-file dedup")

        train_recs, val_recs, cal_recs, test_recs = train_val_cal_test_split(
            all_records, seed=args.seed
        )
        for name, recs in (
            ("train", train_recs),
            ("val", val_recs),
            ("cal", cal_recs),
            ("test", test_recs),
        ):
            split_records[name] = recs
            split_hashes[name] = {
                sha1(f"{r['token']}|{r['line_content']}".encode()).hexdigest()
                for r in recs
            }
            split_tokens[name] = {r["token"] for r in recs}
            if verbose:
                print(f"  {name}: {len(recs)} records")
    else:
        if verbose:
            print(f"Loading training data from {args.train_data} …")
        recs, hashes, tokens = load_records(args.train_data, deduplicate=True)
        split_records["train"] = recs
        split_hashes["train"] = hashes
        split_tokens["train"] = tokens
        if verbose:
            print(f"  train: {len(recs)} records")

        for name, path in (
            ("val", args.val_data),
            ("cal", args.cal_data),
            ("test", args.test_data),
        ):
            if path is None:
                continue
            if verbose:
                print(f"Loading {name} data from {path} …")
            recs, hashes, tokens = load_records(path, deduplicate=True)
            split_records[name] = recs
            split_hashes[name] = hashes
            split_tokens[name] = tokens
            if verbose:
                print(f"  {name}: {len(recs)} records")

        if "cal" not in split_records and "val" in split_records:
            print(
                "\nWARNING: No --cal-data provided. Calibration will reuse --val-data "
                "(same split used for early stopping). Use --cal-data for a clean pipeline."
            )

    # ------------------------------------------------------------------ #
    # Phase 2: Cross-split overlap protection.                            #
    # ------------------------------------------------------------------ #
    if len(split_records) > 1:
        if verbose:
            print("\nRunning cross-split overlap check …")
        split_records = check_cross_split_overlap(
            split_records, split_hashes, split_tokens,
            max_drop_pct=args.max_overlap, verbose=verbose,
        )

    # ------------------------------------------------------------------ #
    # Phase 3: Extract features from filtered records.                    #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nExtracting features …")
    X_train, y_train = extract_features_from_records(split_records["train"])
    if verbose:
        print(f"  train: {len(X_train)} samples ({N_FEATURES} features)")

    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    if "val" in split_records:
        X_val, y_val = extract_features_from_records(split_records["val"])

    X_cal: Optional[np.ndarray] = None
    y_cal: Optional[np.ndarray] = None
    if "cal" in split_records:
        X_cal, y_cal = extract_features_from_records(split_records["cal"])

    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    if "test" in split_records:
        X_test, y_test = extract_features_from_records(split_records["test"])

    # ------------------------------------------------------------------ #
    # Phase 4: Load golden data — quarantined, bypasses phases 1–3.       #
    # ------------------------------------------------------------------ #
    X_golden: Optional[np.ndarray] = None
    y_golden: Optional[np.ndarray] = None
    if args.golden_data:
        if verbose:
            print(f"\nLoading golden OOD data from {args.golden_data} …")
        golden_recs, _, _ = load_records(args.golden_data, deduplicate=False)
        X_golden, y_golden = extract_features_from_records(golden_recs)
        if verbose:
            print(f"  golden: {len(X_golden)} samples (quarantined)")

    # ------------------------------------------------------------------ #
    # Phase 5: Train.                                                      #
    # ------------------------------------------------------------------ #
    model, raw_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, seed=args.seed, verbose=verbose
    )

    feature_importance_top10 = report_feature_importance(model, verbose=verbose)

    # ------------------------------------------------------------------ #
    # Phase 6: Calibrate and find thresholds.                             #
    # ------------------------------------------------------------------ #
    threshold_low = 0.15
    threshold_high = 0.85
    optimal_threshold = 0.5
    platt_a = 0.0
    platt_b = 0.0
    calibrator = None

    # Resolve calibration data: prefer dedicated cal split, fall back to val.
    X_calib = X_cal if X_cal is not None else X_val
    y_calib = y_cal if y_cal is not None else y_val

    if X_calib is not None and y_calib is not None:
        calibrator = calibrate_model(model, X_calib, y_calib, verbose=verbose)
        cal_proba = calibrator.predict_proba(X_calib)[:, 1]
        threshold_low, threshold_high, optimal_threshold = find_dual_thresholds(
            y_calib,
            cal_proba,
            target_recall=args.target_recall,
            target_precision=args.target_precision,
            verbose=verbose,
        )
        platt_estimator = calibrator.calibrated_classifiers_[0]
        platt_a = float(platt_estimator.calibrators[0].a_)
        platt_b = float(platt_estimator.calibrators[0].b_)
    else:
        if verbose:
            print(
                "\nWARNING: No calibration data — skipping Platt scaling and "
                "threshold search. Using default thresholds."
            )

    # ------------------------------------------------------------------ #
    # Phase 7: Evaluate on synthetic test set, then golden OOD.          #
    # ------------------------------------------------------------------ #
    eval_metrics: Dict[str, Any] = {}
    golden_metrics: Dict[str, Any] = {}

    if calibrator is not None:
        # Synthetic evaluation: prefer dedicated test split, fall back to cal.
        X_eval = X_test if X_test is not None else X_calib
        y_eval = y_test if y_test is not None else y_calib
        eval_label = (
            "SYNTHETIC TEST"
            if X_test is not None
            else "CALIBRATION (no --test-data provided)"
        )

        if X_eval is not None and y_eval is not None:
            eval_metrics = evaluate(
                calibrator,
                threshold_low,
                threshold_high,
                X_eval,
                y_eval,
                label=eval_label,
                verbose=verbose,
            )

        # Golden OOD evaluation — same calibrator, same thresholds.
        if X_golden is not None and y_golden is not None:
            golden_metrics = evaluate(
                calibrator,
                threshold_low,
                threshold_high,
                X_golden,
                y_golden,
                label="GOLDEN OOD",
                verbose=verbose,
            )

    # ------------------------------------------------------------------ #
    # Phase 8: Save model and config.                                      #
    # ------------------------------------------------------------------ #
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
        "golden_metrics": golden_metrics,
        "training_samples": len(X_train),
        "validation_samples": len(X_val) if X_val is not None else 0,
        "calibration_samples": len(X_calib) if X_calib is not None else 0,
        "test_samples": len(X_test) if X_test is not None else 0,
        "golden_samples": len(X_golden) if X_golden is not None else 0,
        "feature_importance_top10": feature_importance_top10,
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
    }

    save_model(model, config, args.output_dir, verbose=verbose)

    # ------------------------------------------------------------------ #
    # Phase 9: Side-by-side summary.                                       #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        if eval_metrics:
            has_golden = bool(golden_metrics)
            header = f"  {'Metric':<18} {'Synthetic Test':>15}"
            if has_golden:
                header += f" {'Golden OOD':>12}"
            print(header)
            print("  " + "-" * (35 + (13 if has_golden else 0)))

            for key, label in (
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1 Score"),
                ("auc_roc", "AUC-ROC"),
                ("review_zone_pct", "Review zone %"),
            ):
                syn_val = eval_metrics.get(key, float("nan"))
                row = f"  {label:<18} {syn_val:>15.4f}"
                if has_golden:
                    gold_val = golden_metrics.get(key, float("nan"))
                    row += f" {gold_val:>12.4f}"
                print(row)
        elif raw_metrics:
            print(f"  Raw AUC-ROC (val): {raw_metrics.get('auc_roc', 'N/A'):.4f}")

        print(f"\n  Threshold low:   {threshold_low:.4f}")
        print(f"  Threshold high:  {threshold_high:.4f}")
        print(f"  Output dir:      {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
