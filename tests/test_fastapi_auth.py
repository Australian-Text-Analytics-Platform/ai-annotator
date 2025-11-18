"""
Test authentication endpoints

Tests API key authentication for protected endpoints.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestAuthentication:
    """Test authentication requirements"""

    def test_valid_api_key_passes_auth(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that valid API key passes authentication"""
        response = client.get("/models/", headers=auth_headers)
        # Should not return 401 or 403
        assert response.status_code != 401
        assert response.status_code != 403

    def test_invalid_api_key_returns_403(
        self,
        client: TestClient,
        invalid_auth_headers: dict
    ):
        """Test that invalid API key returns 403 Forbidden"""
        response = client.get("/models/", headers=invalid_auth_headers)
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_missing_api_key_returns_403(self, client: TestClient):
        """Test that missing API key returns 403"""
        response = client.get("/models/")
        # FastAPI with auto_error=True returns 403 for missing header
        assert response.status_code == 403

    def test_health_endpoint_bypasses_auth(self, client: TestClient):
        """Test that health endpoint doesn't require authentication"""
        response = client.get("/health/")
        assert response.status_code == 200

    def test_root_endpoint_no_auth(self, client: TestClient):
        """Test that root endpoint doesn't require authentication"""
        response = client.get("/")
        assert response.status_code == 200


class TestProtectedEndpoints:
    """Test that endpoints requiring auth are properly protected"""

    def test_models_endpoint_requires_auth(self, client: TestClient):
        """Test /models/ requires authentication"""
        response = client.get("/models/")
        assert response.status_code == 403

    def test_classify_batch_requires_auth(
        self,
        client: TestClient,
        sample_classification_request: dict
    ):
        """Test /classify/batch requires authentication"""
        response = client.post("/classify/batch", json=sample_classification_request)
        assert response.status_code == 403

    def test_estimate_cost_requires_auth(
        self,
        client: TestClient,
        sample_cost_estimate_request: dict
    ):
        """Test /classify/estimate-cost requires authentication"""
        response = client.post("/classify/estimate-cost", json=sample_cost_estimate_request)
        assert response.status_code == 403

    def test_jobs_endpoint_requires_auth(self, client: TestClient):
        """Test /jobs/{job_id} requires authentication"""
        response = client.get("/jobs/test-job-id")
        assert response.status_code == 403

    def test_cancel_job_requires_auth(self, client: TestClient):
        """Test DELETE /jobs/{job_id} requires authentication"""
        response = client.delete("/jobs/test-job-id")
        assert response.status_code == 403


class TestAuthHeaderFormat:
    """Test authentication header format"""

    def test_correct_header_name(self, client: TestClient, auth_headers: dict):
        """Test that X-API-Key header is used"""
        assert "X-API-Key" in auth_headers

        response = client.get("/models/", headers=auth_headers)
        assert response.status_code != 403

    def test_wrong_header_name_fails(self, client: TestClient):
        """Test that wrong header name fails authentication"""
        wrong_headers = {"Authorization": "Bearer test-key"}
        response = client.get("/models/", headers=wrong_headers)
        assert response.status_code == 403

    def test_case_sensitive_header_name(self, client: TestClient):
        """Test that header name is case-insensitive (HTTP standard)"""
        # HTTP headers are case-insensitive
        headers = {"x-api-key": "test-service-key-12345"}
        response = client.get("/models/", headers=headers)
        # Should work with lowercase
        assert response.status_code != 403


@pytest.mark.asyncio
class TestAuthenticationAsync:
    """Async authentication tests"""

    async def test_valid_api_key_async(
        self,
        async_client: AsyncClient,
        auth_headers: dict
    ):
        """Test valid API key with async client"""
        response = await async_client.get("/models/", headers=auth_headers)
        assert response.status_code != 403

    async def test_invalid_api_key_async(
        self,
        async_client: AsyncClient,
        invalid_auth_headers: dict
    ):
        """Test invalid API key with async client"""
        response = await async_client.get("/models/", headers=invalid_auth_headers)
        assert response.status_code == 403

    async def test_missing_api_key_async(self, async_client: AsyncClient):
        """Test missing API key with async client"""
        response = await async_client.get("/models/")
        assert response.status_code == 403


class TestAuthService:
    """Test AuthService directly"""

    def test_auth_service_validates_correct_key(self, test_auth_service):
        """Test that auth service validates correct key"""
        from tests.conftest import TEST_SERVICE_API_KEY

        assert TEST_SERVICE_API_KEY in test_auth_service.valid_api_keys

    def test_auth_service_rejects_invalid_key(self, test_auth_service):
        """Test that auth service rejects invalid key"""
        from tests.conftest import INVALID_API_KEY

        assert INVALID_API_KEY not in test_auth_service.valid_api_keys

    def test_auth_service_with_multiple_keys(self):
        """Test auth service with multiple valid keys"""
        from classifier_fastapi.auth import AuthService

        keys = {"key1", "key2", "key3"}
        auth_service = AuthService(valid_api_keys=keys)

        assert "key1" in auth_service.valid_api_keys
        assert "key2" in auth_service.valid_api_keys
        assert "key3" in auth_service.valid_api_keys
        assert "invalid" not in auth_service.valid_api_keys
