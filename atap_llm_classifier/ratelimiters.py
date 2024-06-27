from collections import namedtuple
from functools import total_ordering, lru_cache

from loguru import logger
from pydantic import BaseModel, computed_field
import asyncio
import contextlib
import enum
from typing import Coroutine, Generator, ContextManager, Self, Union

__all__ = [
    "RateLimit",
    "RateLimiterAlg",
    "RateLimiters",
    "TokenBucket",
]

from atap_llm_classifier import config
from atap_llm_classifier.providers import LLMModelProperties, LLMProvider
from atap_llm_classifier.utils import utils


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
    def __init__(self, request: "TokenBucket", tokens: Union["TokenBucket", None]):
        self._request: TokenBucket = request
        self._tokens: TokenBucket = tokens

        self._a_stack = contextlib.AsyncExitStack()

    @property
    def request(self) -> "TokenBucket":
        return self._request

    @property
    def tokens(self) -> Union["TokenBucket", None]:
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


ProviderRateLimits = namedtuple("ProviderRateLimits", ["requests", "tokens"])


def get_rate_limiters(
    model_props: LLMModelProperties,
) -> RateLimiters:
    rate_limits: ProviderRateLimits = get_rate_limits(model_props)
    rlimiter_reqs = config.rate_limiter_alg.make_rate_limiter(rate_limits.requests)
    rlimiter_toks = config.rate_limiter_alg.make_rate_limiter(rate_limits.tokens)
    return RateLimiters(rlimiter_reqs, rlimiter_toks)


@lru_cache
def get_rate_limits(
    model_props: LLMModelProperties,
) -> ProviderRateLimits:
    return ProviderRateLimits(
        requests=get_rate_limit_for_requests(model_props),
        tokens=get_rate_limit_for_tokens(model_props),
    )


@lru_cache
def get_rate_limit_for_requests(
    model_props: LLMModelProperties,
) -> RateLimit:
    candidates: list[RateLimit] = list()
    if not config.mock.enabled:
        match model_props.provider:
            case LLMProvider.OPENAI:
                if not model_props.is_authenticated():
                    raise ValueError("Provided model props is not authenticated.")
                model, api_key = model_props.name, model_props.api_key
                candidates.append(
                    _get_openai_rate_limit(model=model, api_key=api_key).requests
                )
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
    if utils.is_jupyter_context():
        logger.info("In jupyter context. Adding jupyter ate limit as a candidate.")
        candidates.append(config.jupyter.requests_rate_limit)

    lowest_rate_limit = sorted(candidates, reverse=False)[0]
    return lowest_rate_limit


def get_rate_limit_for_tokens(
    model_props: LLMModelProperties,
) -> RateLimit | None:
    if not config.mock.enabled:
        match model_props.provider:
            case LLMProvider.OPENAI:
                if not model_props.is_authenticated():
                    raise ValueError("Provided model props is not authenticated.")
                return _get_openai_rate_limit(
                    model_props.model, model_props.api_key
                ).tokens
            case _:
                pass  # todo: rate limit for other providers - currently goes to default.
    else:
        return config.mock.tokens_rate_limit
    return None


@lru_cache
def _get_openai_rate_limit(
    model: str,
    api_key: str,
) -> ProviderRateLimits:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.with_raw_response.create(
            messages=[
                {
                    "role": "user",
                    "content": "test",
                }
            ],
            model=model,
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
