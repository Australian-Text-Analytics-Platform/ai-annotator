from pydantic import BaseModel

from atap_llm_classifier.ratelimiters import RateLimit, RateLimiterAlg


class _MockConfig(BaseModel):
    enabled: bool
    requests_rate_limit: RateLimit | None
    tokens_rate_limit: RateLimit | None


class _JupyterConfig(BaseModel):
    requests_rate_limit: RateLimit | None


class _BatchConfig(BaseModel):
    num_workers: int
    rate_limit_max_retries: int
    rate_limit_retry_exp_backoff_first_wait_s: float


mock: _MockConfig = _MockConfig(
    enabled=False,
    requests_rate_limit=RateLimit(
        max_requests=100,
        per_seconds=1.0,
    ),
    tokens_rate_limit=RateLimit(
        max_requests=20_000,
        per_seconds=30.0,
    ),
)

jupyter: _JupyterConfig = _JupyterConfig(
    requests_rate_limit=RateLimit(
        max_requests=100,
        per_seconds=1.0,
    )
)

batch: _BatchConfig = _BatchConfig(
    num_workers=5,
    rate_limit_max_retries=10,
    rate_limit_retry_exp_backoff_first_wait_s=3.0,
)

rate_limiter_alg: RateLimiterAlg = RateLimiterAlg.TOKEN_BUCKET
default_requests_rate_limit: RateLimit | None = RateLimit(
    max_requests=100, per_seconds=1.0
)
default_tokens_rate_limit: RateLimit | None = None
