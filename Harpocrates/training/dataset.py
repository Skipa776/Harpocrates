"""
Dataset loading and validation for Harpocrates ML training.

Provides utilities for loading JSONL datasets, validating schema,
extracting features, and splitting into train/val/test sets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from Harpocrates.ml.features import extract_features_from_record

# Required fields in training records
REQUIRED_FIELDS = ["token", "line_content", "context_before", "context_after", "label"]

# Optional fields
OPTIONAL_FIELDS = ["file_path", "label_reason", "secret_type", "negative_type"]


class DatasetError(Exception):
    """Exception raised for dataset validation errors."""

    pass


def validate_record(record: Dict[str, Any], line_num: int = 0) -> List[str]:
    """
    Validate a single training record.

    Args:
        record: Dict to validate
        line_num: Line number for error messages

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Line {line_num}: Missing required field '{field}'")

    # Validate field types
    if "token" in record and not isinstance(record["token"], str):
        errors.append(f"Line {line_num}: 'token' must be a string")

    if "line_content" in record and not isinstance(record["line_content"], str):
        errors.append(f"Line {line_num}: 'line_content' must be a string")

    if "context_before" in record and not isinstance(record["context_before"], list):
        errors.append(f"Line {line_num}: 'context_before' must be a list")

    if "context_after" in record and not isinstance(record["context_after"], list):
        errors.append(f"Line {line_num}: 'context_after' must be a list")

    if "label" in record:
        if not isinstance(record["label"], int) or record["label"] not in (0, 1):
            errors.append(f"Line {line_num}: 'label' must be 0 or 1")

    return errors


def load_jsonl(
    path: Path,
    validate: bool = True,
    skip_invalid: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load training data from JSONL file.

    Args:
        path: Path to JSONL file
        validate: If True, validate each record
        skip_invalid: If True, skip invalid records instead of raising

    Returns:
        List of training records

    Raises:
        DatasetError: If validation fails and skip_invalid is False
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    all_errors = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                error = f"Line {line_num}: Invalid JSON - {e}"
                if skip_invalid:
                    all_errors.append(error)
                    continue
                raise DatasetError(error) from e

            if validate:
                errors = validate_record(record, line_num)
                if errors:
                    if skip_invalid:
                        all_errors.extend(errors)
                        continue
                    raise DatasetError("\n".join(errors))

            records.append(record)

    if all_errors:
        print(f"Warning: Skipped {len(all_errors)} invalid records")
        for error in all_errors[:5]:
            print(f"  {error}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more")

    return records


def iter_jsonl(path: Path, validate: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Iterate over JSONL file (streaming, memory-efficient).

    Args:
        path: Path to JSONL file
        validate: If True, validate each record

    Yields:
        Training records
    """
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if validate:
                errors = validate_record(record, line_num)
                if errors:
                    raise DatasetError("\n".join(errors))

            yield record


def extract_features_batch(
    records: List[Dict[str, Any]],
) -> Tuple[List[List[float]], List[int]]:
    """
    Extract features from a batch of records.

    Args:
        records: List of training records

    Returns:
        Tuple of (feature_arrays, labels)
    """
    features_list = []
    labels = []

    for record in records:
        features = extract_features_from_record(record)
        features_list.append(features.to_array())
        labels.append(record["label"])

    return features_list, labels


def train_val_test_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split records into train/val/test sets.

    Args:
        records: List of training records
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test = 1 - train - val)
        seed: Random seed for reproducibility
        stratify: If True, maintain label balance in splits

    Returns:
        Tuple of (train_records, val_records, test_records)
    """
    import random

    if seed is not None:
        random.seed(seed)

    if stratify:
        # Separate by label
        positive = [r for r in records if r["label"] == 1]
        negative = [r for r in records if r["label"] == 0]

        random.shuffle(positive)
        random.shuffle(negative)

        # Split each class
        def split_list(lst: List, train_r: float, val_r: float):
            train_end = int(len(lst) * train_r)
            val_end = int(len(lst) * (train_r + val_r))
            return lst[:train_end], lst[train_end:val_end], lst[val_end:]

        pos_train, pos_val, pos_test = split_list(positive, train_ratio, val_ratio)
        neg_train, neg_val, neg_test = split_list(negative, train_ratio, val_ratio)

        # Combine and shuffle
        train = pos_train + neg_train
        val = pos_val + neg_val
        test = pos_test + neg_test

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return train, val, test
    else:
        # Simple random split
        shuffled = records.copy()
        random.shuffle(shuffled)

        train_end = int(len(shuffled) * train_ratio)
        val_end = int(len(shuffled) * (train_ratio + val_ratio))

        return (
            shuffled[:train_end],
            shuffled[train_end:val_end],
            shuffled[val_end:],
        )


def get_label_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get distribution of labels in dataset."""
    positive = sum(1 for r in records if r["label"] == 1)
    negative = len(records) - positive
    return {
        "positive": positive,
        "negative": negative,
        "total": len(records),
        "positive_ratio": positive / len(records) if records else 0,
    }


def get_type_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get distribution of secret/negative types in dataset."""
    types: Dict[str, int] = {}

    for record in records:
        if record["label"] == 1:
            key = record.get("secret_type", "unknown_positive")
        else:
            key = record.get("negative_type", "unknown_negative")

        types[key] = types.get(key, 0) + 1

    return types


class Dataset:
    """
    Dataset wrapper for training data.

    Provides convenient access to features, labels, and metadata.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        name: str = "dataset",
    ):
        """
        Initialize dataset.

        Args:
            records: List of training records
            name: Dataset name for logging
        """
        self.records = records
        self.name = name
        self._features: Optional[List[List[float]]] = None
        self._labels: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    @property
    def features(self) -> List[List[float]]:
        """Extract features (cached)."""
        if self._features is None:
            self._features, self._labels = extract_features_batch(self.records)
        return self._features

    @property
    def labels(self) -> List[int]:
        """Get labels (cached)."""
        if self._labels is None:
            self._features, self._labels = extract_features_batch(self.records)
        return self._labels

    @property
    def label_distribution(self) -> Dict[str, int]:
        """Get label distribution."""
        return get_label_distribution(self.records)

    @property
    def type_distribution(self) -> Dict[str, int]:
        """Get type distribution."""
        return get_type_distribution(self.records)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        name: Optional[str] = None,
        validate: bool = True,
    ) -> "Dataset":
        """
        Load dataset from JSONL file.

        Args:
            path: Path to JSONL file
            name: Dataset name (defaults to filename)
            validate: If True, validate records

        Returns:
            Dataset instance
        """
        records = load_jsonl(path, validate=validate)
        return cls(records, name=name or path.stem)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """
        Split into train/val/test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train, val, test = train_val_test_split(
            self.records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        return (
            Dataset(train, name=f"{self.name}_train"),
            Dataset(val, name=f"{self.name}_val"),
            Dataset(test, name=f"{self.name}_test"),
        )

    def summary(self) -> str:
        """Get dataset summary string."""
        dist = self.label_distribution
        return (
            f"Dataset: {self.name}\n"
            f"  Total: {dist['total']}\n"
            f"  Positive: {dist['positive']} ({dist['positive_ratio']:.1%})\n"
            f"  Negative: {dist['negative']} ({1 - dist['positive_ratio']:.1%})"
        )


__all__ = [
    "Dataset",
    "DatasetError",
    "load_jsonl",
    "iter_jsonl",
    "validate_record",
    "extract_features_batch",
    "train_val_test_split",
    "get_label_distribution",
    "get_type_distribution",
]
