from functools import total_ordering

from pydantic import BaseModel, computed_field
import asyncio
import contextlib
import enum
from typing import Coroutine, Generator, ContextManager, Self, Iterable

__all__ = [
    "RateLimit",
    "RateLimiterAlg",
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

    def make_rate_limiter(self, rate_limit: RateLimit) -> "TokenBucket":
        match self:
            case RateLimiterAlg.TOKEN_BUCKET:
                return TokenBucket.from_rate_limit(rate_limit)


class RateLimiters(object):
    def __init__(self, request: "TokenBucket", tokens: "TokenBucket" | None):
        self._request: TokenBucket = request
        self._tokens: TokenBucket = tokens

        self._a_stack = contextlib.AsyncExitStack()

    @property
    def request(self) -> "TokenBucket":
        return self._request

    @property
    def tokens(self) -> "TokenBucket" | None:
        return self._tokens

    def acquire_for_request(self, tokens: int):
        return self._request.acquire(tokens)

    def acquire_for_tokens(self, tokens: int):
        return self._tokens.acquire(tokens)

    async def __aenter__(self):
        with self._a_stack as stack:
            for rlimiter in [self._request, self._tokens]:
                stack.enter_async_context(rlimiter)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._a_stack.__aexit__(exc_type, exc_val, exc_tb)


class TokenBucket(object):
    @classmethod
    def from_rate_limit(cls, rate_limit: RateLimit) -> Self:
        return cls(
            capacity=rate_limit.max_requests,
            replenish_rate_s=rate_limit.per_seconds,
        )

    def __init__(self, capacity: int, replenish_rate_s: float):
        self._capacity = capacity
        self._remaining = capacity
        self._replenish_rate_s = replenish_rate_s

        self._cond = asyncio.Condition()
        self._replenisher: asyncio.Task | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def remaining(self) -> int:
        return self._remaining

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
            self._remaining = min(self._remaining + tokens, self._capacity)
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
                    await asyncio.sleep(delay=self._replenish_rate_s)

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

    async def __aenter__(self):
        if not self.replenisher_running:
            self.start_replenisher()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


# todo: archive this
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
