import asyncio
import contextlib
import enum
from typing import Coroutine, Generator, Callable, ContextManager

from pydantic import BaseModel

__all__ = [
    "RateLimiter",
    "RateLimit",
]


class RateLimit(BaseModel):
    max_requests: int
    per_seconds: float


class RateLimiter(enum.Enum):
    TOKEN_BUCKET: str = "token_bucket"

    def get_context_manager(
        self,
    ) -> Callable[..., ContextManager[Generator[Coroutine, None, None]]]:
        match self:
            case RateLimiter.TOKEN_BUCKET:
                return token_bucket


@contextlib.contextmanager
def token_bucket(
    on: list[Coroutine],
    max_requests: int,
    per_seconds: float,
) -> ContextManager[Generator[Coroutine, None, None]]:
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
