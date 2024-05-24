import tempfile

from pydantic import Field
from pydantic_settings import BaseSettings

from atap_llm_classifier.techniques.parsers import OutputFormat

__all__ = [
    "Settings",
]


class Settings(BaseSettings):
    SEED: int | None = None  # reproducible
    LLM_OUTPUT_FORMAT: OutputFormat

    USE_MOCK: bool = False
    CHECKPOINT_DIR: str = Field(
        default=tempfile.mkdtemp(), description="Default checkpoint directory."
    )
