from pydantic import Field
from pydantic_settings import BaseSettings

__all__ = ["settings"]


class APISettings(BaseSettings):
    OPENAI_API_KEY: str = Field(min_length=10, max_length=10)


settings = APISettings()
