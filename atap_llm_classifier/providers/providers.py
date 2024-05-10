"""providers.py"""

from enum import Enum

from pydantic import BaseModel, Field, HttpUrl

from atap_llm_classifier.assets import Asset

__all__ = [
    "LLMProvider",
    "LLMProviderContext",
]


class LLMProviderContext(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    privacy_policy_url: HttpUrl | None = Field(default=None, frozen=True)


class LLMProvider(Enum):
    OPENAI: str = "openai"
    OPENAI_AZURE_SIH: str = "openai_azure_sih"

    def get_context(self):
        match self:
            case LLMProvider.OPENAI:
                return LLMProviderContext(**Asset.PROVIDERS.get(self.value))
            case LLMProvider.OPENAI_AZURE_SIH:
                return LLMProviderContext(**Asset.PROVIDERS.get(self.value))
