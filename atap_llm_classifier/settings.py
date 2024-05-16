from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SEED: int | None = None  # reproducible

    USE_MOCK: bool = False
