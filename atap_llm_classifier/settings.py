from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    USE_MOCK: bool = False
