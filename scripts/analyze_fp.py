#!/usr/bin/env python3
"""
Analyze false positives to understand precision gap.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from Harpocrates.ml.features import FeatureVector, extract_features_from_record

    # Load data
    data_path = Path("Harpocrates/training/data/training_data_v2.pkl")
    with open(data_path, "rb") as f:
        records = pickle.load(f)

    # Extract features
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

    X = np.array(features)
    y = np.array(labels)

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    val_idx = indices[split:]

    X_val = X[val_idx]
    y_val = y[val_idx]
    val_records = [valid_records[i] for i in val_idx]

    # Train a quick model
    import lightgbm as lgb

    train_idx = indices[:split]
    X_train = X[train_idx]
    y_train = y[train_idx]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        scale_pos_weight=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba > 0.4).astype(int)

    # Find false positives
    fp_mask = (y_pred == 1) & (y_val == 0)
    fp_indices = np.where(fp_mask)[0]

    print(f"Total samples: {len(y_val)}")
    print(f"Predicted positive: {sum(y_pred)}")
    print(f"False positives: {len(fp_indices)}")
    print()

    # Analyze FP records
    print("=" * 60)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 60)

    # Group by semantic type
    fp_types = Counter()
    fp_var_names = Counter()
    fp_tokens = []

    for idx in fp_indices:
        record = val_records[idx]
        fp_types[record.get("type", "unknown")] += 1
        fp_var_names[record.get("var_name", "unknown")] += 1
        fp_tokens.append(record.get("token", "")[:40])

    print("\nFP by type:")
    for t, count in fp_types.most_common(15):
        print(f"  {t}: {count}")

    print("\nFP by variable name (top 20):")
    for name, count in fp_var_names.most_common(20):
        print(f"  {name}: {count}")

    print("\nSample FP tokens:")
    for token in fp_tokens[:10]:
        print(f"  {token}")

    # Feature analysis
    feature_names = FeatureVector.get_feature_names()

    # Get TP samples
    tp_mask = (y_pred == 1) & (y_val == 1)

    if sum(fp_mask) > 0 and sum(tp_mask) > 0:
        fp_means = X_val[fp_mask].mean(axis=0)
        tp_means = X_val[tp_mask].mean(axis=0)

        print("\n" + "=" * 60)
        print("FEATURE DIFFERENCES (FP vs TP)")
        print("=" * 60)

        diffs = []
        for i, name in enumerate(feature_names):
            diff = fp_means[i] - tp_means[i]
            diffs.append((name, diff, fp_means[i], tp_means[i]))

        diffs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{'Feature':<30} {'FP Mean':>12} {'TP Mean':>12} {'Diff':>12}")
        print("-" * 70)
        for name, diff, fp_m, tp_m in diffs[:20]:
            print(f"{name:<30} {fp_m:>12.4f} {tp_m:>12.4f} {diff:>+12.4f}")

    # Look at high-confidence false positives
    high_conf_fp = (y_proba > 0.7) & (y_val == 0)
    if sum(high_conf_fp) > 0:
        print("\n" + "=" * 60)
        print(f"HIGH-CONFIDENCE FALSE POSITIVES (prob > 0.7): {sum(high_conf_fp)}")
        print("=" * 60)

        hc_indices = np.where(high_conf_fp)[0][:10]
        for idx in hc_indices:
            record = val_records[idx]
            print(f"\nType: {record.get('type')}")
            print(f"Var: {record.get('var_name')}")
            print(f"Token: {record.get('token', '')[:60]}")
            print(f"Prob: {y_proba[idx]:.3f}")


if __name__ == "__main__":
    main()
