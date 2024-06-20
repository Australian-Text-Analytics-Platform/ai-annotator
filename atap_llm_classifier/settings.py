from functools import lru_cache
from collections import namedtuple

from loguru import logger
from pydantic_settings import BaseSettings

from atap_llm_classifier import config
from atap_llm_classifier.formatter.models import OutputFormat
from atap_llm_classifier.providers import (
    LLMProvider,
    LLMModelUserProperties,
)
from atap_llm_classifier.ratelimiters import (
    RateLimit,
    RateLimiterAlg,
    RateLimiters,
)
from atap_llm_classifier.utils.utils import is_jupyter_context

__all__ = [
    "get_settings",
    "get_rate_limits",
    "ProviderRateLimits",
]


class Settings(BaseSettings):
    LLM_OUTPUT_FORMAT: OutputFormat = OutputFormat.YAML

    BATCH_NUM_WORKERS: int = 5
    RATE_LIMITER_ALG: RateLimiterAlg = RateLimiterAlg.TOKEN_BUCKET
    BASE_REQUESTS_RATE_LIMIT: RateLimit = RateLimit(
        max_requests=100,
        per_seconds=1.0,
    )
    # note: on how to override RATE_LIMIT using env vars:
    #   BASE_REQUESTS_RATE_LIMIT__MAX_REQUESTS=
    #   BASE_REQUESTS_RATE_LIMIT__PER_SECONDS=
    JUPYTER_REQUESTS_RATE_LIMIT: RateLimit = RateLimit(
        max_requests=100, per_seconds=1.0
    )
    BATCH_RATE_LIMIT_MAX_RETRIES: int = 10
    BATCH_RATE_LIMIT_RETRY_EXP_BACKOFF_FIRST_WAIT_S: float = 3.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


ProviderRateLimits = namedtuple("ProviderRateLimits", ["requests", "tokens"])


def get_rate_limiters(
    user_model: LLMModelUserProperties,
) -> RateLimiters:
    rate_limits: ProviderRateLimits = get_rate_limits(user_model)
    rlimiter_reqs = get_settings().RATE_LIMITER_ALG.make_rate_limiter(
        rate_limits.requests
    )
    rlimiter_toks = get_settings().RATE_LIMITER_ALG.make_rate_limiter(
        rate_limits.tokens
    )
    return RateLimiters([rlimiter_reqs, rlimiter_toks])


@lru_cache
def get_rate_limits(
    user_model: LLMModelUserProperties,
) -> ProviderRateLimits:
    return ProviderRateLimits(
        requests=get_rate_limit_for_requests(user_model),
        tokens=get_rate_limit_for_tokens(user_model),
    )


@lru_cache
def get_rate_limit_for_requests(
    user_model: LLMModelUserProperties,
) -> RateLimit:
    candidates: list[RateLimit] = list()
    if not config.mock:
        match user_model.provider:
            case LLMProvider.OPENAI:
                candidates.append(_get_openai_rate_limit(user_model).requests)
            case _:
                pass  # todo: rate limit for other providers - currently goes to default.
    else:
        if config.mock_requests_rate_limit is not None:
            candidates.append(config.mock_requests_rate_limit)

    if len(candidates) < 1:
        logger.info(
            "No rate limits from provider found. Adding default as base rate limit in candidates."
        )
        candidates.append(get_settings().BASE_REQUESTS_RATE_LIMIT)
    if is_jupyter_context():
        logger.info("In jupyter context. Adding jupyter ate limit as a candidate.")
        candidates.append(get_settings().JUPYTER_REQUESTS_RATE_LIMIT)

    lowest_rate_limit = sorted(candidates, reverse=False)[0]
    return lowest_rate_limit


def get_rate_limit_for_tokens(
    user_model: LLMModelUserProperties,
) -> RateLimit | None:
    if not config.mock:
        match user_model.provider:
            case LLMProvider.OPENAI:
                return _get_openai_rate_limit(user_model).tokens
            case _:
                pass  # todo: rate limit for other providers - currently goes to default.
    else:
        return config.mock_tokens_rate_limit


@lru_cache
def _get_openai_rate_limit(
    user_model_props: LLMModelUserProperties,
) -> ProviderRateLimits:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=user_model_props.validated_api_key.get_secret_value())
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
