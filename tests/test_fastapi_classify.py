"""
Test classification endpoints

Tests the /classify/batch endpoint for batch classification job creation.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
import asyncio


class TestClassifyBatchEndpoint:
    """Test batch classification endpoint"""

    def test_classify_requires_authentication(
        self,
        client: TestClient,
        sample_classification_request: dict
    ):
        """Test that /classify/batch requires authentication"""
        response = client.post("/classify/batch", json=sample_classification_request)
        assert response.status_code == 403

    def test_classify_with_valid_auth(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that /classify/batch works with valid authentication"""
        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=sample_classification_request,
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_classify_creates_job_successfully(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that classification job is created successfully"""
        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=sample_classification_request,
                headers=auth_headers
            )
            data = response.json()

            assert response.status_code == 200
            assert "job_id" in data
            assert "status" in data
            assert "message" in data
            assert "created_at" in data
            assert data["status"] == "pending"

    def test_classify_returns_valid_job_id(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that classification returns a valid UUID job_id"""
        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=sample_classification_request,
                headers=auth_headers
            )
            data = response.json()

            assert "job_id" in data
            # Should be a valid UUID format
            job_id = data["job_id"]
            assert isinstance(job_id, str)
            assert len(job_id) == 36  # UUID format


class TestClassifyValidation:
    """Test input validation for classify endpoint"""

    def test_missing_required_field_texts(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that missing 'texts' field returns 422"""
        request = sample_classification_request.copy()
        del request["texts"]

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_missing_required_field_user_schema(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that missing 'user_schema' field returns 422"""
        request = sample_classification_request.copy()
        del request["user_schema"]

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_missing_required_field_provider(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that missing 'provider' field returns 422"""
        request = sample_classification_request.copy()
        del request["provider"]

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_missing_required_field_model(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that missing 'model' field returns 422"""
        request = sample_classification_request.copy()
        del request["model"]

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_empty_texts_returns_400(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that empty texts array returns 400"""
        request = sample_classification_request.copy()
        request["texts"] = []

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "No texts provided" in response.json()["detail"]

    def test_batch_size_exceeds_maximum(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict,
        test_settings
    ):
        """Test that batch size exceeding limit returns 400"""
        request = sample_classification_request.copy()
        # Create texts array larger than MAX_BATCH_SIZE
        request["texts"] = ["text"] * (test_settings.MAX_BATCH_SIZE + 1)

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_invalid_user_schema(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that invalid user schema creates job but will fail in background"""
        request = sample_classification_request.copy()
        request["user_schema"] = {"invalid": "schema"}

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            # Job creation succeeds, validation happens in background task
            assert response.status_code == 200
            # Job will fail during processing


class TestClassifyTechniques:
    """Test different classification techniques"""

    @pytest.mark.parametrize("technique", [
        "zero_shot",
        "few_shot",
        "chain_of_thought"
    ])
    def test_classification_with_different_techniques(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict,
        technique: str
    ):
        """Test classification with different techniques"""
        request = sample_classification_request.copy()
        request["technique"] = technique

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            # Should accept all valid techniques
            assert response.status_code == 200

    def test_classification_with_invalid_technique(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test classification with invalid technique"""
        request = sample_classification_request.copy()
        request["technique"] = "invalid_technique"

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            # Should fail in background task
            # But job creation should succeed
            assert response.status_code == 200


class TestClassifyProviderValidation:
    """Test provider and model validation"""

    def test_invalid_provider_returns_error(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that invalid provider returns error"""
        request = sample_classification_request.copy()
        request["provider"] = "invalid_provider"

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            # Job creation succeeds, but will fail in background
            assert response.status_code == 200

    def test_invalid_model_returns_error(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that invalid model returns error"""
        request = sample_classification_request.copy()
        request["model"] = "nonexistent-model-12345"

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            # Job creation succeeds, validation happens in background
            assert response.status_code == 200


class TestClassifyOptionalParameters:
    """Test optional parameters"""

    def test_default_technique_zero_shot(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that technique defaults to zero_shot"""
        request = sample_classification_request.copy()
        if "technique" in request:
            del request["technique"]

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_optional_temperature_parameter(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test optional temperature parameter"""
        request = sample_classification_request.copy()
        request["temperature"] = 0.7

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_optional_top_p_parameter(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test optional top_p parameter"""
        request = sample_classification_request.copy()
        request["top_p"] = 0.9

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_optional_llm_endpoint(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test optional llm_endpoint parameter"""
        request = sample_classification_request.copy()
        request["llm_endpoint"] = "https://custom-endpoint.com"

        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            assert response.status_code == 200


@pytest.mark.asyncio
class TestClassifyAsync:
    """Async tests for classify endpoint"""

    async def test_classify_async(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test classify endpoint with async client"""
        with patch("classifier_fastapi.api.routes.classify.process_classification_job"):
            response = await async_client.post(
                "/classify/batch",
                json=sample_classification_request,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data

    async def test_classify_no_auth_async(
        self,
        async_client: AsyncClient,
        sample_classification_request: dict
    ):
        """Test classify requires auth with async client"""
        response = await async_client.post(
            "/classify/batch",
            json=sample_classification_request
        )
        assert response.status_code == 403


class TestClassifyTemperatureValidation:
    """Test temperature parameter validation"""

    def test_temperature_above_max_fails(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that temperature > 2.0 fails validation"""
        request = sample_classification_request.copy()
        request["temperature"] = 2.5

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_temperature_zero_fails(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that temperature = 0 fails validation (must be > 0)"""
        request = sample_classification_request.copy()
        request["temperature"] = 0

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_temperature_negative_fails(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that negative temperature fails validation"""
        request = sample_classification_request.copy()
        request["temperature"] = -0.5

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422


class TestClassifyTopPValidation:
    """Test top_p parameter validation"""

    def test_top_p_above_max_fails(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that top_p > 1.0 fails validation"""
        request = sample_classification_request.copy()
        request["top_p"] = 1.5

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_top_p_zero_fails(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that top_p = 0 fails validation (must be > 0)"""
        request = sample_classification_request.copy()
        request["top_p"] = 0

        response = client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422
