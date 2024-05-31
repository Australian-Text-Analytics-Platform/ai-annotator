from pydantic import BaseModel
import asyncio
import contextlib
import enum
from typing import Coroutine, Generator, Callable, ContextManager

from atap_llm_classifier.providers import (
    LLMProviderUserProperties,
    LLMUserModelProperties,
)

__all__ = [
    "RateLimit",
    "RateLimiterAlg",
    "get_openai_rate_limit",
]


class RateLimit(BaseModel):
    max_requests: int
    per_seconds: float


class RateLimiterAlg(enum.Enum):
    TOKEN_BUCKET: str = "token_bucket"

    def get_rate_limiter(
        self,
    ) -> Callable[
        ..., ContextManager[tuple[Generator[Coroutine, None, None], asyncio.Semaphore]]
    ]:
        match self:
            case RateLimiterAlg.TOKEN_BUCKET:
                return token_bucket


@contextlib.contextmanager
def token_bucket(
    on: list[Coroutine],
    rate_limit: RateLimit,
) -> ContextManager[tuple[Generator[Coroutine, None, None], asyncio.Semaphore]]:
    max_requests, per_seconds = rate_limit.max_requests, rate_limit.per_seconds

    sem = asyncio.Semaphore(max_requests)

    async def replenish_tokens():
        while True:
            await asyncio.sleep(per_seconds)
            for i in range(max_requests):
                sem.release()

    async def rate_limited(coro: Coroutine):
        async with sem:
            return await coro

    replenisher = asyncio.create_task(replenish_tokens())
    try:
        yield [rate_limited(coro=coro) for coro in on], sem
    except Exception as e:
        raise e
    finally:
        replenisher.cancel()


def get_openai_rate_limit(
    user_model_props: LLMUserModelProperties,
) -> RateLimit:
    from openai import OpenAI

    client = OpenAI(api_key=user_model_props.validated_api_key.get_secret_value())
    resp = client.completions.with_raw_response.create(
        messages=[
            {
                "role": "user",
                "content": "test",
            }
        ],
        model=user_model_props.name,
    )
    max_requests = int(resp.headers["x-ratelimit-limit-requests"])
    per_seconds: str = resp.headers["x-ratelimit-reset-requests"].strip()
    if per_seconds.endswith("ms"):
        per_seconds: float = float(per_seconds[:-2]) / 1000
    elif per_seconds.endswith("s"):
        per_seconds: float = float(per_seconds[:-1])
    else:
        raise Exception(
            "OpenAI rate limit reset expected to end with either 's' or 'ms'."
        )
    return RateLimit(max_requests=max_requests, per_seconds=per_seconds)
