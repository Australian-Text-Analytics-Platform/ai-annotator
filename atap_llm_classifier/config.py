from atap_llm_classifier.ratelimiters import RateLimit

mock: bool = False
mock_requests_rate_limit: RateLimit | None = RateLimit(
    max_requests=100, per_seconds=1.0
)
mock_tokens_rate_limit: RateLimit | None = RateLimit(
    max_requests=20_000,
    per_seconds=30,
)
