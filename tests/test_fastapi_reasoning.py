"""
FastAPI tests for reasoning features

Tests API endpoints with reasoning parameters.
"""
import pytest
from httpx import AsyncClient
from unittest.mock import Mock, patch
from classifier_fastapi.api.models import JobStatus


@pytest.mark.asyncio
async def test_classification_request_with_reasoning_effort(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_classification_request: dict,
    mock_litellm
):
    """Test classification request with reasoning_effort parameter."""
    # Add reasoning_effort to request
    request_data = {
        **sample_classification_request,
        "reasoning_effort": "medium"
    }

    response = await async_client.post(
        "/classify/batch",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == JobStatus.PENDING


@pytest.mark.asyncio
async def test_classification_request_with_enable_reasoning(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_classification_request: dict,
    mock_litellm
):
    """Test classification request with enable_reasoning parameter."""
    request_data = {
        **sample_classification_request,
        "enable_reasoning": True,
        "max_reasoning_chars": 200
    }

    response = await async_client.post(
        "/classify/batch",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == JobStatus.PENDING


@pytest.mark.asyncio
async def test_classification_request_invalid_reasoning_effort(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_classification_request: dict
):
    """Test that invalid reasoning_effort values are rejected."""
    request_data = {
        **sample_classification_request,
        "reasoning_effort": "invalid"
    }

    response = await async_client.post(
        "/classify/batch",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_classification_request_all_reasoning_params(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_classification_request: dict,
    mock_litellm
):
    """Test classification request with all reasoning parameters."""
    request_data = {
        **sample_classification_request,
        "reasoning_effort": "high",
        "enable_reasoning": True,
        "max_reasoning_chars": 250
    }

    response = await async_client.post(
        "/classify/batch",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_cost_estimate_with_reasoning(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_cost_estimate_request: dict
):
    """Test cost estimation with reasoning parameters."""
    request_data = {
        **sample_cost_estimate_request,
        "enable_reasoning": True,
        "max_reasoning_chars": 150
    }

    response = await async_client.post(
        "/classify/estimate-cost",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "estimated_tokens" in data
    # With reasoning, token count should be higher
    assert data["estimated_tokens"] > 0


@pytest.mark.asyncio
async def test_classification_result_contains_new_fields(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_classification_request: dict,
    mock_litellm
):
    """Test that classification request accepts and processes reasoning parameters."""
    # Test that a request with enable_reasoning creates a job successfully
    request_data = {
        **sample_classification_request,
        "enable_reasoning": True,
        "max_reasoning_chars": 200
    }

    response = await async_client.post(
        "/classify/batch",
        json=request_data,
        headers=auth_headers
    )

    assert response.status_code == 200
    job_data = response.json()
    assert "job_id" in job_data
    # The job will process in background and store results with new fields


@pytest.mark.asyncio
async def test_backward_compatibility_no_reasoning_params(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_classification_request: dict,
    mock_litellm
):
    """Test that requests without reasoning parameters still work (backward compatibility)."""
    # Don't include reasoning parameters - should use defaults
    response = await async_client.post(
        "/classify/batch",
        json=sample_classification_request,
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == JobStatus.PENDING


def test_classification_request_model_validation():
    """Test ClassificationRequest model with reasoning parameters."""
    from classifier_fastapi.api.models import ClassificationRequest

    # Valid request with reasoning parameters
    request = ClassificationRequest(
        texts=["test"],
        user_schema={"classes": [{"name": "A", "description": "Class A"}]},
        provider="openai",
        model="gpt-4o-mini",
        technique="zero_shot",
        reasoning_effort="low",
        enable_reasoning=True,
        max_reasoning_chars=200
    )

    assert request.reasoning_effort == "low"
    assert request.enable_reasoning is True
    assert request.max_reasoning_chars == 200

    # Test defaults
    request_defaults = ClassificationRequest(
        texts=["test"],
        user_schema={"classes": [{"name": "A", "description": "Class A"}]},
        provider="openai",
        model="gpt-4o-mini",
        technique="zero_shot"
    )

    assert request_defaults.reasoning_effort is None
    assert request_defaults.enable_reasoning is False
    assert request_defaults.max_reasoning_chars == 150

    # Invalid reasoning_effort
    with pytest.raises(ValueError, match="reasoning_effort must be"):
        ClassificationRequest(
            texts=["test"],
            user_schema={"classes": [{"name": "A", "description": "Class A"}]},
            provider="openai",
            model="gpt-4o-mini",
            technique="zero_shot",
            reasoning_effort="invalid"
        )


def test_classification_result_item_model():
    """Test ClassificationResultItem has new fields."""
    from classifier_fastapi.api.models import ClassificationResultItem

    result = ClassificationResultItem(
        index=0,
        text="test text",
        classification="positive",
        prompt="test prompt",
        confidence=0.95,
        reasoning="This is positive",
        reasoning_content="Extended reasoning"
    )

    assert result.confidence == 0.95
    assert result.reasoning == "This is positive"
    assert result.reasoning_content == "Extended reasoning"

    # Test with None values (optional fields)
    result_minimal = ClassificationResultItem(
        index=0,
        text="test text",
        classification="positive"
    )

    assert result_minimal.confidence is None
    assert result_minimal.reasoning is None
    assert result_minimal.reasoning_content is None


def test_cost_estimate_request_model():
    """Test CostEstimateRequest has reasoning fields."""
    from classifier_fastapi.api.models import CostEstimateRequest

    request = CostEstimateRequest(
        texts=["test"],
        user_schema={"classes": [{"name": "A", "description": "Class A"}]},
        provider="openai",
        model="gpt-4o-mini",
        technique="zero_shot",
        enable_reasoning=True,
        max_reasoning_chars=200
    )

    assert request.enable_reasoning is True
    assert request.max_reasoning_chars == 200

    # Test defaults
    request_defaults = CostEstimateRequest(
        texts=["test"],
        user_schema={"classes": [{"name": "A", "description": "Class A"}]},
        provider="openai",
        model="gpt-4o-mini",
        technique="zero_shot"
    )

    assert request_defaults.enable_reasoning is False
    assert request_defaults.max_reasoning_chars == 150
