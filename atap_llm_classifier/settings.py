import tempfile
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from atap_llm_classifier.formatter.models import OutputFormat

__all__ = [
    "Settings",
    "get_settings",
]


class Settings(BaseSettings):
    SEED: int | None = None  # reproducible
    LLM_OUTPUT_FORMAT: OutputFormat

    USE_MOCK: bool = False
    CHECKPOINT_DIR: str = Field(
        default=tempfile.mkdtemp(), description="Default checkpoint directory."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # todo: fixed here for dev purposese.
    return Settings(
        USE_MOCK=True,
        LLM_OUTPUT_FORMAT="yaml",
    )
