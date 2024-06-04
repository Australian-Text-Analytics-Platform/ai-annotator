import tempfile
from functools import lru_cache

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings

from atap_llm_classifier import config
from atap_llm_classifier.formatter.models import OutputFormat
from atap_llm_classifier.providers import (
    LLMProviderUserProperties,
    LLMProvider,
    LLMUserModelProperties,
)
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
    DEFAULT_RATE_LIMIT: RateLimit = RateLimit(max_requests=100, per_seconds=1.0)
    JUPYTER_RATE_LIMIT: RateLimit = RateLimit(max_requests=100, per_seconds=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_rate_limit(
    user_model_props: LLMUserModelProperties,
) -> RateLimit:
    # todo: return the lowest rate limit.
    candidates: list[RateLimit] = list()
    if not config.mock:
        match user_model_props.provider:
            case LLMProvider.OPENAI:
                try:
                    candidates.append(get_openai_rate_limit(user_model_props))
                except Exception as e:
                    logger.warning(f"Unable to retrieve OpenAI rate limit. Err: {e}")
                    logger.warning(
                        "OpenAI rate limit will not be added as one of the rate limit candidates."
                    )
            case _:
                # todo: rate limit for other providers
                pass
    if len(candidates) < 1:
        logger.info(
            "No rate limits from provider found. Adding default rate limit as a candidate."
        )
        candidates.append(get_settings().DEFAULT_RATE_LIMIT)
    if is_jupyter_context():
        logger.info("In jupyter context. Adding jupyter ate limit as a candidate.")
        candidates.append(get_settings().JUPYTER_RATE_LIMIT)

    lowest_rate_limit = sorted(candidates, reverse=False)[0]
    return lowest_rate_limit
