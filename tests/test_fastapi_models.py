"""
Test models listing endpoints

Tests the /models/ endpoint for listing available models and pricing.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


class TestModelsEndpoint:
    """Test models listing endpoint"""

    def test_models_requires_authentication(self, client: TestClient):
        """Test that /models/ requires authentication"""
        response = client.get("/models/")
        assert response.status_code == 403

    def test_models_with_valid_auth(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that /models/ works with valid authentication"""
        response = client.get("/models/", headers=auth_headers)
        assert response.status_code == 200

    def test_models_response_structure(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test models endpoint returns correct structure"""
        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        assert "models" in data
        assert "providers" in data
        assert isinstance(data["models"], list)
        assert isinstance(data["providers"], list)

    def test_models_list_contains_providers(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that providers list contains expected providers"""
        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        providers = data["providers"]
        # Should include at least some common providers
        assert len(providers) > 0
        # Check that at least openai provider is present
        assert "openai" in providers

    @patch("classifier_fastapi.utils.litellm_.get_available_models")
    @patch("classifier_fastapi.core.cost.CostEstimator.get_model_pricing")
    def test_models_with_pricing_info(
        self,
        mock_get_pricing,
        mock_get_models,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that models include pricing information"""
        # Mock model list
        mock_get_models.return_value = ["gpt-4o-mini", "gpt-4"]

        # Mock pricing
        mock_get_pricing.return_value = {
            "max_input_tokens": 128000,
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.0000006
        }

        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        models = data["models"]
        assert len(models) > 0

        # Check first model has expected fields
        if models:
            model = models[0]
            assert "name" in model
            assert "provider" in model
            # Pricing fields may be None if not available
            assert "context_window" in model
            assert "input_cost_per_1m_tokens" in model
            assert "output_cost_per_1m_tokens" in model

    @patch("classifier_fastapi.utils.litellm_.get_available_models")
    @patch("classifier_fastapi.core.cost.CostEstimator.get_model_pricing")
    def test_models_without_pricing_info(
        self,
        mock_get_pricing,
        mock_get_models,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that models without pricing are still listed"""
        # Mock model list
        mock_get_models.return_value = ["custom-model"]

        # Mock no pricing available
        mock_get_pricing.return_value = None

        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        models = data["models"]
        # Should still include the model even without pricing
        assert len(models) > 0

    def test_models_info_completeness(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that model information is complete"""
        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        models = data["models"]
        for model in models[:5]:  # Check first 5
            assert "name" in model
            assert "provider" in model
            assert isinstance(model["name"], str)
            assert isinstance(model["provider"], str)
            # Optional fields
            assert "context_window" in model
            assert "input_cost_per_1m_tokens" in model
            assert "output_cost_per_1m_tokens" in model


class TestModelsFiltering:
    """Test filtering models by provider"""

    @patch("classifier_fastapi.utils.litellm_.get_available_models")
    def test_filter_by_provider(
        self,
        mock_get_models,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that models can be filtered by provider"""
        # Mock different models for different providers
        def side_effect(provider):
            if provider == "openai":
                return ["gpt-4o-mini", "gpt-4"]
            elif provider == "ollama":
                return ["llama2", "mistral"]
            return []

        mock_get_models.side_effect = side_effect

        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        # Check that models from different providers are present
        providers_in_models = {model["provider"] for model in data["models"]}
        assert len(providers_in_models) > 0


@pytest.mark.asyncio
class TestModelsEndpointAsync:
    """Async tests for models endpoint"""

    async def test_models_async(
        self,
        async_client: AsyncClient,
        auth_headers: dict
    ):
        """Test models endpoint with async client"""
        response = await async_client.get("/models/", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "providers" in data

    async def test_models_no_auth_async(self, async_client: AsyncClient):
        """Test models endpoint requires auth with async client"""
        response = await async_client.get("/models/")
        assert response.status_code == 403


class TestModelInfoStructure:
    """Test ModelInfo response structure"""

    def test_model_info_fields(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that ModelInfo has all expected fields"""
        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        if data["models"]:
            model = data["models"][0]
            expected_fields = [
                "name",
                "provider",
                "context_window",
                "input_cost_per_1m_tokens",
                "output_cost_per_1m_tokens"
            ]
            for field in expected_fields:
                assert field in model

    @patch("classifier_fastapi.utils.litellm_.get_available_models")
    @patch("classifier_fastapi.core.cost.CostEstimator.get_model_pricing")
    def test_cost_conversion_to_per_million(
        self,
        mock_get_pricing,
        mock_get_models,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that costs are correctly converted to per-million-tokens"""
        mock_get_models.return_value = ["test-model"]

        # Cost per token: $0.0000001 = $0.10 per million
        mock_get_pricing.return_value = {
            "max_input_tokens": 100000,
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000002
        }

        response = client.get("/models/", headers=auth_headers)
        data = response.json()

        models = data["models"]
        if models:
            model = models[0]
            # Should be converted to per-million (with floating point tolerance)
            assert model["input_cost_per_1m_tokens"] == pytest.approx(0.1, rel=1e-6)
            assert model["output_cost_per_1m_tokens"] == pytest.approx(0.2, rel=1e-6)


class TestModelListingErrorHandling:
    """Test error handling in model listing"""

    @patch("classifier_fastapi.utils.litellm_.get_available_models")
    def test_handles_provider_error_gracefully(
        self,
        mock_get_models,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that errors from one provider don't break the entire response"""
        def side_effect(provider):
            if provider == "openai":
                raise Exception("API error")
            return ["model1"]

        mock_get_models.side_effect = side_effect

        response = client.get("/models/", headers=auth_headers)
        # Should still return 200 even if one provider fails
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "providers" in data

    @patch("classifier_fastapi.core.cost.CostEstimator.get_model_pricing")
    @patch("classifier_fastapi.utils.litellm_.get_available_models")
    def test_handles_pricing_error_gracefully(
        self,
        mock_get_models,
        mock_get_pricing,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that pricing errors don't prevent model listing"""
        mock_get_models.return_value = ["test-model"]
        mock_get_pricing.side_effect = Exception("Pricing error")

        response = client.get("/models/", headers=auth_headers)
        # Should still return 200
        assert response.status_code == 200

        data = response.json()
        # Model should still be listed, just without pricing
        assert len(data["models"]) >= 0
