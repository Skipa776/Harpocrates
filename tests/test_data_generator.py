"""
Tests for synthetic training data generation.
"""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from Harpocrates.training.generators.context_templates import (
    generate_negative_context,
    generate_positive_context,
)
from Harpocrates.training.generators.generate_data import (
    generate_training_data,
    generate_training_record,
)
from Harpocrates.training.generators.secret_templates import (
    generate_aws_key,
    generate_checksum,
    generate_git_sha,
    generate_github_token,
    generate_jwt_token,
    generate_openai_key,
    generate_uuid,
)


class TestSecretTemplates:
    """Tests for secret token generation."""

    def test_aws_key_format(self):
        """Test AWS key format is correct."""
        access_key, secret_key = generate_aws_key()

        assert access_key.startswith("AKIA")
        assert len(access_key) == 20
        assert len(secret_key) == 40

    def test_github_token_format(self):
        """Test GitHub token format is correct."""
        token = generate_github_token()

        assert token.startswith("ghp_")
        assert len(token) == 40  # ghp_ + 36

    def test_jwt_token_format(self):
        """Test JWT token has valid structure."""
        token = generate_jwt_token()

        parts = token.split(".")
        assert len(parts) == 3  # header.payload.signature
        assert parts[0].startswith("eyJ")  # Base64 encoded JSON

    def test_openai_key_format(self):
        """Test OpenAI key format is correct."""
        token = generate_openai_key()

        assert token.startswith("sk-")
        assert len(token) == 51  # sk- + 48

    def test_git_sha_format(self):
        """Test Git SHA format is correct."""
        sha = generate_git_sha()

        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_uuid_format(self):
        """Test UUID format is correct."""
        uuid = generate_uuid()

        parts = uuid.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert parts[2].startswith("4")  # Version 4

    def test_checksum_formats(self):
        """Test checksum generation for different algorithms."""
        md5 = generate_checksum("md5")
        sha256 = generate_checksum("sha256")

        # Either with or without prefix
        assert len(md5.replace("md5:", "")) == 32
        assert len(sha256.replace("sha256:", "")) == 64


class TestContextTemplates:
    """Tests for context generation."""

    def test_positive_context_structure(self):
        """Test positive context has required structure."""
        token = "AKIAIOSFODNN7EXAMPLE"
        line, before, after, file_path = generate_positive_context(
            token=token,
            var_name="api_key",
            language="python",
        )

        assert token in line
        assert "api_key" in line
        assert isinstance(before, list)
        assert isinstance(after, list)
        assert file_path.endswith(".py")

    def test_negative_context_git_sha(self):
        """Test negative context for Git SHA."""
        sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        line, before, after, file_path = generate_negative_context(
            token=sha,
            var_name="commit_sha",
            negative_type="safe",
        )

        assert sha in line
        assert "commit_sha" in line or "sha" in line.lower()

    def test_test_context(self):
        """Test that test context includes test indicators."""
        token = "fake_secret_123"
        line, before, after, file_path = generate_negative_context(
            token=token,
            negative_type="test",
        )

        # Should be in a test file path
        assert "test" in file_path.lower()


class TestTrainingRecordGeneration:
    """Tests for training record generation."""

    def test_positive_record_structure(self):
        """Test positive record has all required fields."""
        record = generate_training_record(positive=True)

        assert "token" in record
        assert "line_content" in record
        assert "context_before" in record
        assert "context_after" in record
        assert "file_path" in record
        assert "label" in record
        assert record["label"] == 1
        # Note: label_reason and secret_type were intentionally removed
        # to prevent label leakage during training

    def test_negative_record_structure(self):
        """Test negative record has all required fields."""
        record = generate_training_record(positive=False)

        assert "token" in record
        assert "line_content" in record
        assert "context_before" in record
        assert "context_after" in record
        assert "file_path" in record
        assert "label" in record
        assert record["label"] == 0
        # Note: label_reason and negative_type were intentionally removed
        # to prevent label leakage during training

    def test_positive_records_have_tokens(self):
        """Test positive records contain non-empty tokens."""
        for _ in range(10):
            record = generate_training_record(positive=True)
            assert record["label"] == 1
            assert len(record["token"]) > 0

    def test_negative_records_have_tokens(self):
        """Test negative records contain non-empty tokens."""
        for _ in range(10):
            record = generate_training_record(positive=False)
            assert record["label"] == 0
            assert len(record["token"]) > 0


class TestDatasetGeneration:
    """Tests for full dataset generation."""

    def test_generate_balanced_dataset(self):
        """Test generating a balanced dataset."""
        records = generate_training_data(count=100, balance=0.5, seed=42)

        assert len(records) == 100

        positive_count = sum(1 for r in records if r["label"] == 1)
        negative_count = len(records) - positive_count

        assert positive_count == 50
        assert negative_count == 50

    def test_generate_imbalanced_dataset(self):
        """Test generating an imbalanced dataset."""
        records = generate_training_data(count=100, balance=0.3, seed=42)

        positive_count = sum(1 for r in records if r["label"] == 1)

        assert positive_count == 30

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        records1 = generate_training_data(count=10, seed=42)
        records2 = generate_training_data(count=10, seed=42)

        for r1, r2 in zip(records1, records2):
            assert r1["token"] == r2["token"]
            assert r1["label"] == r2["label"]

    def test_record_is_jsonl_serializable(self):
        """Test that records can be serialized to JSONL."""
        records = generate_training_data(count=10, seed=42)

        for record in records:
            # Should not raise
            json_str = json.dumps(record)
            parsed = json.loads(json_str)
            assert parsed["token"] == record["token"]


class TestDatasetLoading:
    """Tests for dataset loading (integration with dataset.py)."""

    def test_load_generated_data(self):
        """Test that generated data can be loaded by dataset module."""
        from Harpocrates.training.dataset import load_jsonl, validate_record

        records = generate_training_data(count=10, seed=42)

        # Write to temp file
        with NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)

        try:
            # Load back
            loaded = load_jsonl(temp_path, validate=True)
            assert len(loaded) == 10

            # Validate each record
            for i, record in enumerate(loaded):
                errors = validate_record(record, i)
                assert len(errors) == 0, f"Validation errors: {errors}"
        finally:
            temp_path.unlink()
