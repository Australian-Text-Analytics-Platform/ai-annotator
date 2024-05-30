import tempfile
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from atap_llm_classifier.formatter.models import OutputFormat
from atap_llm_classifier.ratelimiters import RateLimiter

__all__ = [
    "get_settings",
]


class Settings(BaseSettings):
    SEED: int | None = None  # reproducible
    LLM_OUTPUT_FORMAT: OutputFormat = OutputFormat.YAML

    USE_MOCK: bool = False
    CHECKPOINT_DIR: str = Field(
        default=tempfile.mkdtemp(), description="Default checkpoint directory."
    )
    RATE_LIMITER: RateLimiter = RateLimiter.TOKEN_BUCKET


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # todo: fixed here for dev purposese.
    # settings: Settings = Settings(
    #     USE_MOCK=True,
    #     LLM_OUTPUT_FORMAT="yaml",
    # )
    # logger.info(f"Settings: {settings}")
    return Settings()
