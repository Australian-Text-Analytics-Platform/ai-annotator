"""
Test cost estimation endpoints

Tests the /classify/estimate-cost endpoint for token and cost estimation.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


class TestCostEstimationEndpoint:
    """Test cost estimation endpoint"""

    def test_estimate_cost_requires_authentication(
        self,
        client: TestClient,
        sample_cost_estimate_request: dict
    ):
        """Test that /classify/estimate-cost requires authentication"""
        response = client.post("/classify/estimate-cost", json=sample_cost_estimate_request)
        assert response.status_code == 403

    def test_estimate_cost_with_valid_auth(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that /classify/estimate-cost works with valid authentication"""
        with patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens") as mock_tokens:
            with patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost") as mock_cost:
                mock_tokens.return_value = {
                    "total_tokens": 1000,
                    "input_tokens": 800,
                    "estimated_output_tokens": 200
                }
                mock_cost.return_value = 0.001

                response = client.post(
                    "/classify/estimate-cost",
                    json=sample_cost_estimate_request,
                    headers=auth_headers
                )
                assert response.status_code == 200

    def test_estimate_cost_response_structure(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost estimation response structure"""
        with patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens") as mock_tokens:
            with patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost") as mock_cost:
                mock_tokens.return_value = {
                    "total_tokens": 1000,
                    "input_tokens": 800,
                    "estimated_output_tokens": 200
                }
                mock_cost.return_value = 0.001

                response = client.post(
                    "/classify/estimate-cost",
                    json=sample_cost_estimate_request,
                    headers=auth_headers
                )
                data = response.json()

                assert "estimated_tokens" in data
                assert "estimated_cost_usd" in data
                assert "provider" in data
                assert "model" in data
                assert "num_texts" in data
                assert "input_tokens" in data
                assert "output_tokens" in data
                assert "warnings" in data


class TestTokenEstimation:
    """Test token estimation for different text lengths"""

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_token_estimation_short_texts(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test token estimation for short texts"""
        mock_tokens.return_value = {
            "total_tokens": 100,
            "input_tokens": 80,
            "estimated_output_tokens": 20
        }
        mock_cost.return_value = 0.0001

        request = sample_cost_estimate_request.copy()
        request["texts"] = ["short text"]

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        data = response.json()

        assert data["num_texts"] == 1
        assert data["estimated_tokens"] == 100

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_token_estimation_long_texts(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test token estimation for long texts"""
        mock_tokens.return_value = {
            "total_tokens": 5000,
            "input_tokens": 4000,
            "estimated_output_tokens": 1000
        }
        mock_cost.return_value = 0.005

        request = sample_cost_estimate_request.copy()
        request["texts"] = ["long text " * 500]  # Very long text

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        data = response.json()

        assert data["estimated_tokens"] == 5000
        assert data["input_tokens"] == 4000
        assert data["output_tokens"] == 1000

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_token_estimation_multiple_texts(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test token estimation for multiple texts"""
        mock_tokens.return_value = {
            "total_tokens": 3000,
            "input_tokens": 2400,
            "estimated_output_tokens": 600
        }
        mock_cost.return_value = 0.003

        request = sample_cost_estimate_request.copy()
        request["texts"] = ["text1", "text2", "text3", "text4", "text5"]

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        data = response.json()

        assert data["num_texts"] == 5


class TestCostCalculation:
    """Test cost calculation for different models"""

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    @patch("classifier_fastapi.core.cost.CostEstimator.get_pricing_per_million")
    def test_cost_calculation_with_pricing(
        self,
        mock_pricing,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost calculation when pricing is available"""
        mock_tokens.return_value = {
            "total_tokens": 1000,
            "input_tokens": 800,
            "estimated_output_tokens": 200
        }
        mock_cost.return_value = 0.0012
        mock_pricing.return_value = {
            "input_per_million": 0.15,
            "output_per_million": 0.60
        }

        response = client.post(
            "/classify/estimate-cost",
            json=sample_cost_estimate_request,
            headers=auth_headers
        )
        data = response.json()

        assert data["estimated_cost_usd"] == 0.0012
        assert data["input_cost_per_1m"] == 0.15
        assert data["output_cost_per_1m"] == 0.60

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_cost_calculation_no_pricing(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost calculation when pricing is not available"""
        mock_tokens.return_value = {
            "total_tokens": 1000,
            "input_tokens": 800,
            "estimated_output_tokens": 200
        }
        mock_cost.return_value = None

        response = client.post(
            "/classify/estimate-cost",
            json=sample_cost_estimate_request,
            headers=auth_headers
        )
        data = response.json()

        assert data["estimated_cost_usd"] is None
        assert len(data["warnings"]) > 0
        assert any("Pricing information not available" in w for w in data["warnings"])


class TestCostEstimationWarnings:
    """Test warnings for unsupported models"""

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_warning_for_unsupported_model(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that warning is returned for unsupported model"""
        mock_tokens.return_value = {
            "total_tokens": 1000,
            "input_tokens": 800,
            "estimated_output_tokens": 200,
            "warning": "Model not in LiteLLM pricing database"
        }
        mock_cost.return_value = None

        response = client.post(
            "/classify/estimate-cost",
            json=sample_cost_estimate_request,
            headers=auth_headers
        )
        data = response.json()

        assert len(data["warnings"]) > 0
        warnings_str = " ".join(data["warnings"])
        assert "not" in warnings_str.lower()

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_multiple_warnings(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that multiple warnings can be returned"""
        mock_tokens.return_value = {
            "total_tokens": 1000,
            "input_tokens": 800,
            "estimated_output_tokens": 200,
            "warning": "Model pricing estimated"
        }
        mock_cost.return_value = None

        response = client.post(
            "/classify/estimate-cost",
            json=sample_cost_estimate_request,
            headers=auth_headers
        )
        data = response.json()

        # Should have warnings from both token estimation and cost calculation
        assert len(data["warnings"]) >= 1


class TestCostEstimationValidation:
    """Test validation for cost estimation requests"""

    def test_missing_required_field_texts(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that missing 'texts' field returns 422"""
        request = sample_cost_estimate_request.copy()
        del request["texts"]

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_missing_required_field_user_schema(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that missing 'user_schema' field returns 422"""
        request = sample_cost_estimate_request.copy()
        del request["user_schema"]

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_invalid_provider(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that invalid provider returns 400"""
        request = sample_cost_estimate_request.copy()
        request["provider"] = "invalid_provider"

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 400

    def test_invalid_model(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that invalid model returns 400"""
        request = sample_cost_estimate_request.copy()
        request["model"] = "nonexistent-model-xyz"

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_invalid_technique(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test that invalid technique returns 400"""
        request = sample_cost_estimate_request.copy()
        request["technique"] = "invalid_technique"

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )
        assert response.status_code == 400


class TestCostEstimationDifferentProviders:
    """Test cost estimation for different providers"""

    @pytest.mark.parametrize("provider,model", [
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-4"),
    ])
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_cost_estimation_different_providers(
        self,
        mock_cost,
        mock_tokens,
        provider: str,
        model: str,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost estimation for different provider/model combinations"""
        mock_tokens.return_value = {
            "total_tokens": 1000,
            "input_tokens": 800,
            "estimated_output_tokens": 200
        }
        mock_cost.return_value = 0.001

        request = sample_cost_estimate_request.copy()
        request["provider"] = provider
        request["model"] = model

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == provider
        assert data["model"] == model


@pytest.mark.asyncio
class TestCostEstimationAsync:
    """Async tests for cost estimation endpoint"""

    async def test_estimate_cost_async(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost estimation with async client"""
        with patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens") as mock_tokens:
            with patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost") as mock_cost:
                mock_tokens.return_value = {
                    "total_tokens": 1000,
                    "input_tokens": 800,
                    "estimated_output_tokens": 200
                }
                mock_cost.return_value = 0.001

                response = await async_client.post(
                    "/classify/estimate-cost",
                    json=sample_cost_estimate_request,
                    headers=auth_headers
                )
                assert response.status_code == 200

    async def test_estimate_cost_no_auth_async(
        self,
        async_client: AsyncClient,
        sample_cost_estimate_request: dict
    ):
        """Test cost estimation requires auth with async client"""
        response = await async_client.post(
            "/classify/estimate-cost",
            json=sample_cost_estimate_request
        )
        assert response.status_code == 403


class TestCostEstimationEdgeCases:
    """Test edge cases for cost estimation"""

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    def test_empty_texts_array(
        self,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost estimation with empty texts array"""
        mock_tokens.return_value = {
            "total_tokens": 0,
            "input_tokens": 0,
            "estimated_output_tokens": 0
        }

        request = sample_cost_estimate_request.copy()
        request["texts"] = []

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )

        # Should handle empty array gracefully
        data = response.json()
        assert data["num_texts"] == 0

    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens")
    @patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost")
    def test_very_large_batch(
        self,
        mock_cost,
        mock_tokens,
        client: TestClient,
        auth_headers: dict,
        sample_cost_estimate_request: dict
    ):
        """Test cost estimation for very large batch"""
        mock_tokens.return_value = {
            "total_tokens": 100000,
            "input_tokens": 80000,
            "estimated_output_tokens": 20000
        }
        mock_cost.return_value = 10.0

        request = sample_cost_estimate_request.copy()
        request["texts"] = ["text"] * 1000

        response = client.post(
            "/classify/estimate-cost",
            json=request,
            headers=auth_headers
        )

        data = response.json()
        assert data["num_texts"] == 1000
        assert data["estimated_tokens"] == 100000
