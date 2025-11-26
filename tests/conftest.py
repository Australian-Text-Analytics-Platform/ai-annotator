"""
Test configuration and fixtures for FastAPI tests

Provides shared fixtures for testing the FastAPI service.
"""
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os

from classifier_fastapi.api.main import app
from classifier_fastapi.settings import Settings
from classifier_fastapi.job_manager import JobManager, get_job_manager
from classifier_fastapi.auth import AuthService, get_auth_service


# Test API Keys
TEST_SERVICE_API_KEY = "test-service-key-12345"
INVALID_API_KEY = "invalid-key"


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with mock API keys"""
    settings = Settings(
        SERVICE_API_KEYS=TEST_SERVICE_API_KEY,
        MAX_BATCH_SIZE=1000,
        MAX_CONCURRENT_JOBS=100,
        DEFAULT_WORKERS=5,
        LOG_LEVEL="INFO"
    )
    return settings


@pytest.fixture
def test_auth_service() -> AuthService:
    """Test authentication service with test API keys"""
    return AuthService(valid_api_keys={TEST_SERVICE_API_KEY})


@pytest.fixture
def test_job_manager() -> JobManager:
    """Test job manager instance"""
    return JobManager(max_jobs=100)


@pytest.fixture
def mock_settings(test_settings):
    """Mock get_settings to return test settings"""
    # Patch all locations where get_settings is imported and used
    with patch("classifier_fastapi.settings.get_settings", return_value=test_settings):
        yield test_settings


@pytest.fixture
def mock_auth_service(test_auth_service):
    """Mock get_auth_service to return test auth service"""
    with patch("classifier_fastapi.auth.get_auth_service", return_value=test_auth_service):
        with patch("classifier_fastapi.api.dependencies.get_auth_service", return_value=test_auth_service):
            yield test_auth_service


@pytest.fixture
def mock_job_manager(test_job_manager):
    """Mock get_job_manager to return test job manager"""
    with patch("classifier_fastapi.job_manager.get_job_manager", return_value=test_job_manager):
        with patch("classifier_fastapi.api.routes.classify.get_job_manager", return_value=test_job_manager):
            with patch("classifier_fastapi.api.routes.jobs.get_job_manager", return_value=test_job_manager):
                yield test_job_manager


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM to avoid real API calls"""
    with patch("litellm.acompletion") as mock_completion:
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="classified_result"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest_asyncio.fixture
async def async_client(
    mock_settings,
    mock_auth_service,
    mock_job_manager
) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing FastAPI endpoints"""
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def client(
    mock_settings,
    mock_auth_service,
    mock_job_manager
) -> Generator[TestClient, None, None]:
    """Synchronous HTTP client for testing FastAPI endpoints"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers() -> dict:
    """Valid authentication headers"""
    return {"X-API-Key": TEST_SERVICE_API_KEY}


@pytest.fixture
def invalid_auth_headers() -> dict:
    """Invalid authentication headers"""
    return {"X-API-Key": INVALID_API_KEY}


@pytest.fixture
def sample_user_schema() -> dict:
    """Sample user schema for testing"""
    return {
        "classes": [
            {"name": "positive", "description": "Positive sentiment"},
            {"name": "negative", "description": "Negative sentiment"},
            {"name": "neutral", "description": "Neutral sentiment"}
        ]
    }


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for classification"""
    return [
        "This is a great product!",
        "I'm not happy with this service.",
        "It's okay, nothing special."
    ]


@pytest.fixture
def sample_classification_request(sample_texts, sample_user_schema) -> dict:
    """Sample classification request payload"""
    return {
        "texts": sample_texts,
        "user_schema": sample_user_schema,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "technique": "zero_shot",
        "modifier": "no_modifier",
        "temperature": 1.0,
        "top_p": 1.0,
        "llm_api_key": os.getenv("OPENAI_API_KEY")
    }


@pytest.fixture
def sample_cost_estimate_request(sample_texts, sample_user_schema) -> dict:
    """Sample cost estimation request payload"""
    return {
        "texts": sample_texts,
        "user_schema": sample_user_schema,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "technique": "zero_shot"
    }


@pytest.fixture(autouse=True)
def reset_job_manager_singleton():
    """Reset job manager singleton between tests"""
    import classifier_fastapi.job_manager
    classifier_fastapi.job_manager._job_manager = None
    yield
    classifier_fastapi.job_manager._job_manager = None


@pytest.fixture(autouse=True)
def clear_provider_cache():
    """Clear LLMProvider cached properties between tests"""
    from classifier_fastapi.providers import LLMProvider
    # Clear the cached_property for each provider
    for provider in LLMProvider:
        if 'properties' in provider.__dict__:
            del provider.__dict__['properties']
    yield
    # Clear again after test
    for provider in LLMProvider:
        if 'properties' in provider.__dict__:
            del provider.__dict__['properties']


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires real API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Set pytest-asyncio default fixture loop scope
pytest_plugins = ('pytest_asyncio',)
