import tempfile

from pydantic import Field
from pydantic_settings import BaseSettings

from atap_llm_classifier.output_formatter import OutputFormat

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

def get_settings() -> Settings:
    return Settings()
