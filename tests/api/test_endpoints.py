"""
Tests for Harpocrates API endpoints.

These tests verify the REST API functionality including:
- Health and config endpoints
- Scan endpoints (single and batch)
- Verify endpoint
- Error handling
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for API."""
    from Harpocrates.api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for /health and /config endpoints."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self, client):
        """Health response should contain all required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "ml_loaded" in data
        assert "uptime_seconds" in data

    def test_health_status_is_healthy(self, client):
        """Health status should be healthy or degraded."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded"]

    def test_config_returns_200(self, client):
        """Config endpoint should return 200."""
        response = client.get("/config")
        assert response.status_code == 200

    def test_config_contains_ml_settings(self, client):
        """Config response should contain ML settings."""
        response = client.get("/config")
        data = response.json()

        assert "ml_enabled" in data
        assert "ml_mode" in data
        assert "model_version" in data


class TestScanEndpoint:
    """Tests for /scan endpoint."""

    def test_scan_empty_content(self, client):
        """Scan with empty content should fail validation."""
        response = client.post("/scan", json={"content": ""})
        # Empty content violates min_length=1
        assert response.status_code == 422

    def test_scan_clean_code(self, client):
        """Scan clean code should return no findings."""
        response = client.post("/scan", json={
            "content": "x = 1 + 2\nprint(x)",
            "ml_verify": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_findings"] == 0

    def test_scan_detects_aws_key(self, client):
        """Scan should detect AWS access key."""
        response = client.post("/scan", json={
            "content": 'aws_key = "AKIAZ3X7ABCDEFGHIJK1"',
            "ml_verify": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_findings"] >= 1
        assert any(f["secret_type"] == "AWS_ACCESS_KEY_ID" for f in data["findings"])

    def test_scan_detects_github_token(self, client):
        """Scan should detect GitHub token patterns."""
        response = client.post("/scan", json={
            "content": 'GITHUB_TOKEN = "ghp_1234567890abcdefGHIJKLMNOPqrstuv12"',
            "ml_verify": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_findings"] >= 1

    def test_scan_with_filename(self, client):
        """Scan should accept optional filename."""
        response = client.post("/scan", json={
            "content": 'key = "secret123"',
            "filename": "config.py",
            "ml_verify": False
        })
        assert response.status_code == 200

    @pytest.mark.skip(reason="Requires ONNX model; blocked until verifier routing fix lands")
    def test_scan_with_ml_verification(self, client):
        """Scan with ML verification should work."""
        response = client.post("/scan", json={
            "content": 'api_key = "AKIAZ3X7ABCDEFGHIJK1"',
            "ml_verify": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["ml_enabled"] is True

    def test_scan_response_fields(self, client):
        """Scan response should have all required fields."""
        response = client.post("/scan", json={
            "content": "x = 1",
            "ml_verify": False
        })
        data = response.json()

        assert "findings" in data
        assert "scan_time_ms" in data
        assert "ml_enabled" in data
        assert "total_findings" in data
        assert "high_confidence_findings" in data

    def test_scan_finding_fields(self, client):
        """Scan findings should have all required fields."""
        response = client.post("/scan", json={
            "content": 'key = "AKIAZ3X7ABCDEFGHIJK1"',
            "ml_verify": False
        })
        data = response.json()

        if data["total_findings"] > 0:
            finding = data["findings"][0]
            assert "secret_type" in finding
            assert "token_redacted" in finding
            assert "severity" in finding
            assert "confidence" in finding
            assert "evidence_type" in finding


class TestBatchScanEndpoint:
    """Tests for /scan/batch endpoint."""

    def test_batch_scan_single_file(self, client):
        """Batch scan with single file should work."""
        response = client.post("/scan/batch", json={
            "files": [
                {"filename": "test.py", "content": "x = 1"}
            ],
            "ml_verify": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 1

    def test_batch_scan_multiple_files(self, client):
        """Batch scan with multiple files should work."""
        response = client.post("/scan/batch", json={
            "files": [
                {"filename": "config.py", "content": 'key = "AKIAZ3X7ABCDEFGHIJK1"'},
                {"filename": "safe.py", "content": "x = 1 + 2"},
                {"filename": "clean.py", "content": "print('hello')"},
            ],
            "ml_verify": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 3
        assert data["total_findings"] >= 1  # At least the AWS key

    def test_batch_scan_empty_files_list(self, client):
        """Batch scan with empty files list should fail."""
        response = client.post("/scan/batch", json={
            "files": [],
            "ml_verify": False
        })
        # Empty list violates min_length=1
        assert response.status_code == 422

    def test_batch_scan_response_fields(self, client):
        """Batch scan response should have all required fields."""
        response = client.post("/scan/batch", json={
            "files": [{"filename": "test.py", "content": "x = 1"}],
            "ml_verify": False
        })
        data = response.json()

        assert "results" in data
        assert "total_files" in data
        assert "total_findings" in data
        assert "scan_time_ms" in data
        assert "files_with_errors" in data


class TestVerifyEndpoint:
    """Tests for /verify endpoint."""

    def test_verify_requires_token(self, client):
        """Verify endpoint requires token parameter."""
        response = client.post("/verify", json={
            "context_before": "key = ",
            "context_after": ""
        })
        assert response.status_code == 422

    @pytest.mark.skip(reason="Requires ONNX model; blocked until verifier routing fix lands")
    def test_verify_with_context(self, client):
        """Verify with context should return prediction."""
        response = client.post("/verify", json={
            "token": "AKIAZ3X7ABCDEFGHIJK1",
            "context_before": 'aws_key = "',
            "context_after": '"',
            "variable_name": "aws_key"
        })
        assert response.status_code == 200
        data = response.json()

        assert "is_secret" in data
        assert "confidence" in data
        assert "decision_path" in data

    @pytest.mark.skip(reason="Requires ONNX model; blocked until verifier routing fix lands")
    def test_verify_minimal_request(self, client):
        """Verify with minimal request should work."""
        response = client.post("/verify", json={
            "token": "secrettoken123"
        })
        assert response.status_code == 200

    @pytest.mark.skip(reason="Requires ONNX model; blocked until verifier routing fix lands")
    def test_verify_confidence_range(self, client):
        """Verify confidence should be between 0 and 1."""
        response = client.post("/verify", json={
            "token": "AKIAZ3X7ABCDEFGHIJK1",
            "context_before": 'key = "'
        })
        data = response.json()

        assert 0.0 <= data["confidence"] <= 1.0


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json(self, client):
        """Invalid JSON should return 422."""
        response = client.post(
            "/scan",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_field(self, client):
        """Missing required field should return 422."""
        response = client.post("/scan", json={})
        assert response.status_code == 422

    def test_invalid_endpoint(self, client):
        """Invalid endpoint should return 404."""
        response = client.get("/invalid")
        assert response.status_code == 404


class TestOpenAPIDocs:
    """Tests for OpenAPI documentation."""

    def test_docs_available(self, client):
        """Swagger UI should be available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """ReDoc should be available."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json_available(self, client):
        """OpenAPI JSON should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
