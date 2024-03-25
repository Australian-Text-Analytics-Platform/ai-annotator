from pydantic import Field
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    OPENAI_API_KEY: str = Field(min_length=51, max_length=51)
