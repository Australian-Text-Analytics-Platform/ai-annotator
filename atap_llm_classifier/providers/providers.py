"""providers.py"""

from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field, HttpUrl

from atap_llm_classifier.assets import Asset

__all__ = [
    "LLMProvider",
    "LLMProviderProperties",
]


class LLMProviderProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    privacy_policy_url: HttpUrl | None = Field(default=None, frozen=True)


class LLMProvider(Enum):
    OPENAI: str = "openai"
    OPENAI_AZURE_SIH: str = "openai_azure_sih"

    @lru_cache
    def get_properties(self):
        match self:
            case LLMProvider.OPENAI:
                return LLMProviderProperties(**Asset.PROVIDERS.get(self.value))
            case LLMProvider.OPENAI_AZURE_SIH:
                return LLMProviderProperties(**Asset.PROVIDERS.get(self.value))
