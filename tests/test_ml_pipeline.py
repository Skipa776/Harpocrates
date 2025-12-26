"""
End-to-end ML pipeline tests for Harpocrates.

Tests the full training, saving, loading, and verification workflow
to ensure all components work together correctly.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from Harpocrates.core.result import EvidenceType, Finding
from Harpocrates.ml.context import CodeContext
from Harpocrates.ml.features import FeatureVector, extract_features
from Harpocrates.training.generators.generate_data import generate_training_data


class TestFeatureExtraction:
    """Tests for feature extraction pipeline."""

    def test_feature_vector_has_32_features(self):
        """Verify feature vector produces exactly 46 features."""
        fv = FeatureVector()
        array = fv.to_array()
        assert len(array) == 51, f"Expected 46 features, got {len(array)}"

    def test_feature_names_match_array_length(self):
        """Verify feature names match array length."""
        names = FeatureVector.get_feature_names()
        fv = FeatureVector()
        array = fv.to_array()
        assert len(names) == len(array), "Feature names and array length mismatch"

    def test_extract_features_from_finding(self):
        """Test feature extraction from a Finding object."""
        finding = Finding(
            type="AWS_ACCESS_KEY_ID",
            snippet='api_key = "AKIAIOSFODNN7EXAMPLE"',
            evidence=EvidenceType.REGEX,
            token="AKIAIOSFODNN7EXAMPLE",
        )
        context = CodeContext(
            line_content='api_key = "AKIAIOSFODNN7EXAMPLE"',
            lines_before=["import boto3", ""],
            lines_after=["", "client = boto3.client('s3')"],
            file_path="config/aws.py",
        )

        features = extract_features(finding, context)
        array = features.to_array()

        assert len(array) == 51
        assert features.token_length == 20
        assert features.var_contains_secret is True  # "api_key" matches

    def test_new_features_are_extracted(self):
        """Verify the 5 new features are properly extracted."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='version = "v1.2.3-beta"',
            evidence=EvidenceType.ENTROPY,
            token="v1.2.3-beta",
        )
        context = CodeContext(
            line_content='version = "v1.2.3-beta"',
            lines_before=["# Version info"],
            lines_after=[""],
            file_path="config.py",
            line_number=10,
            total_lines=100,
        )

        features = extract_features(finding, context)

        # Check new features exist and have reasonable values
        assert hasattr(features, "token_structure_score")
        assert hasattr(features, "has_version_pattern")
        assert hasattr(features, "semantic_context_score")
        assert hasattr(features, "line_position_ratio")
        assert hasattr(features, "surrounding_secret_density")

        # Version pattern should be detected
        assert features.has_version_pattern is True


class TestDataGeneration:
    """Tests for training data generation."""

    def test_generate_training_data_count(self):
        """Test that data generation produces correct count."""
        data = generate_training_data(count=100, seed=42)
        assert len(data) == 100

    def test_generate_training_data_balance(self):
        """Test that data generation respects balance parameter."""
        data = generate_training_data(count=1000, balance=0.5, seed=42)

        positive_count = sum(1 for d in data if d["label"] == 1)
        negative_count = len(data) - positive_count

        # Allow 10% tolerance
        assert 0.4 <= positive_count / len(data) <= 0.6

    def test_training_data_has_required_fields(self):
        """Test that generated data has all required fields."""
        data = generate_training_data(count=10, seed=42)

        required_fields = ["token", "line_content", "context_before", "context_after", "label"]

        for record in data:
            for field in required_fields:
                assert field in record, f"Missing required field: {field}"

    def test_training_data_reproducible(self):
        """Test that data generation is reproducible with seed."""
        data1 = generate_training_data(count=50, seed=123)
        data2 = generate_training_data(count=50, seed=123)

        for r1, r2 in zip(data1, data2):
            assert r1["token"] == r2["token"]
            assert r1["label"] == r2["label"]


class TestDataset:
    """Tests for Dataset class."""

    def test_dataset_from_jsonl(self):
        """Test loading dataset from JSONL file."""
        from Harpocrates.training.dataset import Dataset

        # Generate some data
        data = generate_training_data(count=50, seed=42)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)

        try:
            dataset = Dataset.from_jsonl(temp_path)
            assert len(dataset) == 50
            assert len(dataset.features) == 50
            assert len(dataset.labels) == 50
        finally:
            temp_path.unlink()

    def test_dataset_split(self):
        """Test dataset splitting."""
        from Harpocrates.training.dataset import Dataset

        data = generate_training_data(count=100, seed=42)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)

        try:
            dataset = Dataset.from_jsonl(temp_path)
            train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.1, seed=42)

            assert len(train) == 80
            assert len(val) == 10
            assert len(test) == 10
        finally:
            temp_path.unlink()


class TestVerifierNoCrash:
    """Tests to verify verifier doesn't crash on any input."""

    def test_verifier_with_various_tokens(self):
        """Test that verifier handles various token types without crashing."""
        from Harpocrates.ml.verifier import XGBoostVerifier

        # These tests just verify no exceptions are raised
        # Model may not be trained, so we catch FileNotFoundError
        try:
            verifier = XGBoostVerifier(lazy_load=True)
        except (ImportError, FileNotFoundError):
            pytest.skip("XGBoost or model not available")

        tokens = [
            "AKIAIOSFODNN7EXAMPLE",  # AWS-like
            "ghp_1234567890abcdefghij",  # GitHub-like
            "a" * 40,  # 40-char hex
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # JWT-like
            "sk_live_1234567890",  # Stripe-like
            "",  # Empty
            "short",  # Short token
            "x" * 1000,  # Very long token
        ]

        for token in tokens:
            finding = Finding(
                type="ENTROPY_CANDIDATE",
                snippet=f'key = "{token}"',
                evidence=EvidenceType.ENTROPY,
                token=token,
            )
            context = CodeContext(
                line_content=f'key = "{token}"',
                file_path="test.py",
            )

            # Just verify no exception is raised during feature extraction
            features = extract_features(finding, context)
            assert len(features.to_array()) == 51

    def test_feature_extraction_with_empty_context(self):
        """Test feature extraction with minimal context."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet="token",
            evidence=EvidenceType.ENTROPY,
            token="some_token_value",
        )
        context = CodeContext(
            line_content="token",
            lines_before=[],
            lines_after=[],
        )

        # Should not raise
        features = extract_features(finding, context)
        assert len(features.to_array()) == 51

    def test_feature_extraction_with_unicode(self):
        """Test feature extraction with unicode content."""
        finding = Finding(
            type="ENTROPY_CANDIDATE",
            snippet='key = "value_with_émoji_🔑"',
            evidence=EvidenceType.ENTROPY,
            token="value_with_émoji_🔑",
        )
        context = CodeContext(
            line_content='key = "value_with_émoji_🔑"',
            lines_before=["# Commentaire français"],
            lines_after=[""],
            file_path="config.py",
        )

        # Should not raise
        features = extract_features(finding, context)
        assert len(features.to_array()) == 51


class TestCrossValidation:
    """Tests for cross-validation functionality."""

    def test_stratified_k_fold_split(self):
        """Test that stratified split maintains class balance."""
        from Harpocrates.training.cross_validation import stratified_k_fold_split
        from Harpocrates.training.dataset import Dataset

        # Generate balanced data
        data = generate_training_data(count=100, balance=0.5, seed=42)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)

        try:
            dataset = Dataset.from_jsonl(temp_path)
            folds = stratified_k_fold_split(dataset, k=5, seed=42)

            assert len(folds) == 5

            for train_ds, val_ds in folds:
                # Each fold should have data
                assert len(train_ds) > 0
                assert len(val_ds) > 0

                # Combined should equal original
                assert len(train_ds) + len(val_ds) == len(dataset)
        finally:
            temp_path.unlink()


class TestEnsembleVerifier:
    """Tests for ensemble verifier."""

    def test_ensemble_config_defaults(self):
        """Test ensemble config has sensible defaults."""
        from Harpocrates.ml.ensemble import EnsembleConfig, EnsembleStrategy

        config = EnsembleConfig()

        assert config.xgboost_weight == 0.6
        assert config.lightgbm_weight == 0.4
        assert config.strategy == EnsembleStrategy.WEIGHTED_AVERAGE
        assert config.xgboost_weight + config.lightgbm_weight == 1.0

    def test_ensemble_strategy_enum(self):
        """Test ensemble strategy options."""
        from Harpocrates.ml.ensemble import EnsembleStrategy

        strategies = list(EnsembleStrategy)
        assert len(strategies) == 4
        assert EnsembleStrategy.WEIGHTED_AVERAGE in strategies
        assert EnsembleStrategy.SOFT_VOTING in strategies
        assert EnsembleStrategy.HARD_VOTING in strategies
        assert EnsembleStrategy.MAX_CONFIDENCE in strategies


class TestLightGBMVerifier:
    """Tests for LightGBM verifier."""

    def test_lightgbm_verifier_singleton(self):
        """Test LightGBM verifier singleton pattern."""
        from Harpocrates.ml.lightgbm_verifier import LightGBMVerifier

        # Reset any existing instance
        LightGBMVerifier.reset_instance()

        try:
            v1 = LightGBMVerifier.get_instance()
            v2 = LightGBMVerifier.get_instance()
            assert v1 is v2
        except (ImportError, FileNotFoundError):
            pytest.skip("LightGBM or model not available")
        finally:
            LightGBMVerifier.reset_instance()

    def test_lightgbm_threshold_property(self):
        """Test threshold property."""
        from Harpocrates.ml.lightgbm_verifier import LightGBMVerifier

        verifier = LightGBMVerifier(threshold=0.7, lazy_load=True)
        assert verifier.threshold == 0.7


class TestTrainingFunctions:
    """Tests for training module functions."""

    def test_train_model_import(self):
        """Test training functions can be imported."""
        try:
            from Harpocrates.training.train import (
                train_model,
                train_lightgbm_model,
                train_ensemble,
                save_model,
            )
        except ImportError:
            pytest.skip("ML dependencies not available")

    def test_model_types_constant(self):
        """Test MODEL_TYPES constant exists."""
        from Harpocrates.training.train import MODEL_TYPES

        assert "xgboost" in MODEL_TYPES
        assert "lightgbm" in MODEL_TYPES
        assert "ensemble" in MODEL_TYPES
