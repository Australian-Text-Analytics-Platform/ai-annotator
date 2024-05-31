import tempfile
from functools import lru_cache

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings

from atap_llm_classifier import config
from atap_llm_classifier.formatter.models import OutputFormat
from atap_llm_classifier.providers import LLMProviderUserProperties, LLMProvider
from atap_llm_classifier.ratelimiters import (
    RateLimit,
    RateLimiterAlg,
    get_openai_rate_limit,
)
from atap_llm_classifier.utils.utils import is_jupyter_context

__all__ = [
    "get_settings",
    "get_rate_limit",
]


class Settings(BaseSettings):
    SEED: int | None = None  # reproducible
    LLM_OUTPUT_FORMAT: OutputFormat = OutputFormat.YAML

    CHECKPOINT_DIR: str = Field(
        default=tempfile.mkdtemp(), description="Default checkpoint directory."
    )
    RATE_LIMITER_ALG: RateLimiterAlg = RateLimiterAlg.TOKEN_BUCKET
    JUPYTER_RATE_LIMIT: RateLimit = RateLimit(max_requests=1000 - 1, per_seconds=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_rate_limit(
    provider_user_props: LLMProviderUserProperties,
    model: str,
) -> RateLimit | None:
    ratelimit: RateLimit | None = None
    if not config.mock:
        match provider_user_props.provider:
            case LLMProvider.OPENAI:
                try:
                    return get_openai_rate_limit(provider_user_props, model)
                except Exception as e:
                    logger.warning(f"Unable to retrieve OpenAI rate limit. Err: {e}")
                    logger.warning("Default rate limit is returned.")
                    ratelimit = None
            case _:
                ratelimit = None
    if is_jupyter_context():
        jup_rate_limit = get_settings().JUPYTER_RATE_LIMIT
        if ratelimit is None:
            return jup_rate_limit
        else:
            if (ratelimit.max_requests / ratelimit.per_seconds) > (
                jup_rate_limit.max_requests / jup_rate_limit.per_seconds
            ):
                logger.warning(
                    f"Detected jupyter context. Rate limit is capped at {jup_rate_limit}."
                )
                return jup_rate_limit
            else:
                return ratelimit
