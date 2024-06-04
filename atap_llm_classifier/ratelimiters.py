from asyncio import Task
from functools import total_ordering

from pydantic import BaseModel, computed_field
import asyncio
import contextlib
import enum
from typing import Coroutine, Generator, Callable, ContextManager, Self

from atap_llm_classifier.providers import (
    LLMProviderUserProperties,
    LLMUserModelProperties,
)

__all__ = [
    "RateLimit",
    "RateLimiterAlg",
    "get_openai_rate_limit",
]


@total_ordering
class RateLimit(BaseModel):
    max_requests: int
    per_seconds: float

    @computed_field
    def max_requests_per_second(self) -> float:
        return self.max_requests / self.per_seconds

    def __eq__(self, other: Self):
        return self.max_requests_per_second == other.max_requests_per_second

    def __gt__(self, other):
        return self.max_requests_per_second > other.max_requests_per_second


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


class TokenBucket(object):
    def __init__(self, capacity: int, rate_ms: int):
        self._capacity = capacity
        self._remaining = capacity
        self._rate_ms = rate_ms

        self._cond = asyncio.Condition()
        self._replenisher: asyncio.Task | None = None

    async def acquire(self, tokens: int) -> int:
        if tokens > self._capacity:
            raise ValueError(
                f"Unable to acquire more tokens than capacity. Requested={tokens} Capacity={self._capacity}."
            )
        async with self._cond:
            while self._remaining < tokens:
                await self._cond.wait()
            self._remaining -= tokens
            return self._remaining

    async def release(self, tokens: int) -> int:
        async with self._cond:
            self._remaining = max(self._remaining + tokens, self._capacity)
            self._cond.notify_all()
            return self._remaining

    @property
    def replenisher_running(self) -> bool:
        return self._replenisher is not None and not self._replenisher.done()

    def start_replenisher(self) -> bool:
        if not self.replenisher_running:

            async def replenisher():
                while True:
                    async with self._cond:
                        self._remaining = self._capacity
                        self._cond.notify_all()
                    await asyncio.sleep(delay=self._rate_ms / 1000)

            self._replenisher = asyncio.create_task(replenisher())
            return self.replenisher_running
        return False

    def cancel_replenisher(self):
        if self._replenisher is not None:
            return self._replenisher.cancel()
        return False

    def destroy(self):
        self.cancel_replenisher()

    def __del__(self):
        self.destroy()


def get_openai_rate_limit(
    user_model_props: LLMUserModelProperties,
) -> RateLimit:
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


if __name__ == "__main__":
    pass
