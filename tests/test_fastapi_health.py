"""
Test health check endpoints

Tests the /health endpoints for service health and readiness checks.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_returns_200(self, client: TestClient):
        """Test that health endpoint returns 200 status"""
        response = client.get("/health/")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client: TestClient):
        """Test health endpoint returns correct response structure"""
        response = client.get("/health/")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_check_status_healthy(self, client: TestClient):
        """Test health endpoint returns 'healthy' status"""
        response = client.get("/health/")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_check_service_info(self, client: TestClient):
        """Test health endpoint returns correct service information"""
        response = client.get("/health/")
        data = response.json()

        assert data["service"] == "ATAP LLM Classifier API"
        assert data["version"] == "0.1.0"

    def test_health_check_uptime(self, client: TestClient):
        """Test health endpoint returns valid uptime"""
        response = client.get("/health/")
        data = response.json()

        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_check_no_auth_required(self, client: TestClient):
        """Test health endpoint is accessible without authentication"""
        # No auth headers
        response = client.get("/health/")
        assert response.status_code == 200

    def test_health_check_with_invalid_auth(self, client: TestClient, invalid_auth_headers: dict):
        """Test health endpoint works even with invalid auth headers"""
        # Health should work regardless of auth headers
        response = client.get("/health/", headers=invalid_auth_headers)
        # Should still work since health doesn't require auth
        assert response.status_code == 200


class TestReadinessEndpoint:
    """Test readiness check endpoint"""

    def test_readiness_check_returns_200(self, client: TestClient):
        """Test that readiness endpoint returns 200 status"""
        response = client.get("/health/ready")
        assert response.status_code == 200

    def test_readiness_check_response(self, client: TestClient):
        """Test readiness endpoint returns correct response"""
        response = client.get("/health/ready")
        data = response.json()

        assert "status" in data
        assert data["status"] == "ready"

    def test_readiness_check_no_auth_required(self, client: TestClient):
        """Test readiness endpoint is accessible without authentication"""
        response = client.get("/health/ready")
        assert response.status_code == 200


@pytest.mark.asyncio
class TestHealthEndpointAsync:
    """Async tests for health endpoint"""

    async def test_health_check_async(self, async_client: AsyncClient):
        """Test health endpoint with async client"""
        response = await async_client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    async def test_readiness_check_async(self, async_client: AsyncClient):
        """Test readiness endpoint with async client"""
        response = await async_client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
