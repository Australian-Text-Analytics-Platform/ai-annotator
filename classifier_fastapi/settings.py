from functools import lru_cache
from collections import namedtuple
from typing import Set

from loguru import logger
from pydantic_settings import BaseSettings

from classifier_fastapi.core import config
from classifier_fastapi.formatter.models import OutputFormat
from classifier_fastapi.providers import (
    LLMProvider,
    LLMModelProperties
)
from classifier_fastapi.ratelimiters import (
    RateLimit,
    RateLimiters,
)
from classifier_fastapi.utils.utils import is_jupyter_context

__all__ = [
    "get_env_settings",
    "get_settings",
    "get_rate_limits",
    "ProviderRateLimits",
]


class Settings(BaseSettings):
    """Application settings"""
    # Service settings
    SERVICE_NAME: str = "ATAP LLM Classifier API"
    SERVICE_VERSION: str = "0.1.0"

    # API authentication
    # Single API key (preferred for simple setups)
    SERVICE_API_KEY: str = ""
    # Comma-separated API keys (for multiple keys, backward compatibility)
    SERVICE_API_KEYS: str = ""

    # Batch configuration
    MAX_BATCH_SIZE: int = 1000
    MAX_CONCURRENT_JOBS: int = 100
    JOB_TIMEOUT_SECONDS: int = 3600

    # Rate limiting
    DEFAULT_WORKERS: int = 5

    # CORS (comma-separated origins)
    CORS_ORIGINS: str = "*"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Output format
    LLM_OUTPUT_FORMAT: OutputFormat = OutputFormat.YAML

    # Ollama settings
    OLLAMA_ENDPOINT: str = "http://127.0.0.1:11434"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env

    @property
    def service_api_keys(self) -> Set[str]:
        """
        Parse API keys from environment variables.
        Supports both SERVICE_API_KEY (single) and SERVICE_API_KEYS (comma-separated).
        """
        keys = set()

        # Add single API key if set
        if self.SERVICE_API_KEY:
            keys.add(self.SERVICE_API_KEY.strip())

        # Add comma-separated keys if set
        if self.SERVICE_API_KEYS:
            keys.update(key.strip() for key in self.SERVICE_API_KEYS.split(",") if key.strip())

        return keys


@lru_cache(maxsize=1)
def get_env_settings() -> Settings:
    """Get environment settings (legacy)"""
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


ProviderRateLimits = namedtuple("ProviderRateLimits", ["requests", "tokens"])


def get_rate_limiters(
    user_model: LLMModelProperties,
) -> RateLimiters:
    rate_limits: ProviderRateLimits = get_rate_limits(user_model)
    rlimiter_reqs = config.rate_limiter_alg.make_rate_limiter(rate_limits.requests)
    rlimiter_toks = config.rate_limiter_alg.make_rate_limiter(rate_limits.tokens)
    return RateLimiters(rlimiter_reqs, rlimiter_toks)


@lru_cache
def get_rate_limits(
    user_model: LLMModelProperties,
) -> ProviderRateLimits:
    return ProviderRateLimits(
        requests=get_rate_limit_for_requests(user_model),
        tokens=get_rate_limit_for_tokens(user_model),
    )


@lru_cache
def get_rate_limit_for_requests(
    user_model: LLMModelProperties,
) -> RateLimit:
    candidates: list[RateLimit] = list()
    if not config.mock.enabled:
        match user_model.provider:
            case LLMProvider.OPENAI:
                candidates.append(_get_openai_rate_limit(user_model).requests)
            case _:
                pass  # todo: rate limit for other providers - currently goes to default.
    else:
        if config.mock.requests_rate_limit is not None:
            candidates.append(config.mock.requests_rate_limit)

    if len(candidates) < 1:
        logger.info(
            "No rate limits from provider found. Adding default as base rate limit in candidates."
        )
        candidates.append(config.default_requests_rate_limit)
    if is_jupyter_context():
        logger.info("In jupyter context. Adding jupyter ate limit as a candidate.")
        candidates.append(config.jupyter.requests_rate_limit)

    lowest_rate_limit = sorted(candidates, reverse=False)[0]
    return lowest_rate_limit


def get_rate_limit_for_tokens(
    user_model: LLMModelProperties,
) -> RateLimit | None:
    if not config.mock.enabled:
        match user_model.provider:
            case LLMProvider.OPENAI:
                return _get_openai_rate_limit(user_model).tokens
            case _:
                pass  # todo: rate limit for other providers - currently goes to default.
    else:
        return config.mock.tokens_rate_limit


@lru_cache
def _get_openai_rate_limit(
    user_model_props: LLMModelProperties,
) -> ProviderRateLimits:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=user_model_props.api_key)
        resp = client.chat.completions.with_raw_response.create(
            messages=[
                {
                    "role": "user",
                    "content": "test",
                }
            ],
            model=user_model_props.name,
        )
    except Exception as e:
        logger.error(f"Failed request rate limits from OpenAI. Err: {e}")
        raise RuntimeError("Failed to retrieve rate limits from OpenAI.") from e
    try:
        max_reqs_per_day = int(resp.headers["x-ratelimit-limit-requests"])
        max_toks_per_min = int(resp.headers["x-ratelimit-limit-tokens"])
    except ValueError as ve:
        logger.error(f"Failed to parse rate limits from OpenAI response. Err: {ve}")
        raise RuntimeError("Failed to retrieve rate limits from OpenAI.") from ve
    except KeyError as ke:
        logger.error(f"No x-ratelimit headers from OpenAI response. Err: {ke}")
        raise RuntimeError("Failed to retrieve rate limits from OpenAI.") from ke
    return ProviderRateLimits(
        requests=RateLimit(max_requests=max_reqs_per_day, per_seconds=60 * 60 * 24),
        tokens=RateLimit(max_requests=max_toks_per_min, per_seconds=60),
    )
