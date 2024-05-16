"""providers.py"""

from enum import Enum
from functools import lru_cache, cached_property

from pydantic import BaseModel, Field, HttpUrl

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.utils import litellm_ as litellm_utils

__all__ = [
    "LLMProvider",
    "LLMProviderProperties",
]


class LLMModelProperties(BaseModel):
    context_window: int | None = None
    description: str | None = None


class LLMProviderProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    privacy_policy_url: HttpUrl | None = Field(default=None, frozen=True)
    models: dict[str, LLMModelProperties]


class LLMProvider(Enum):
    OPENAI: str = "openai"
    OPENAI_AZURE_SIH: str = "openai_azure_sih"

    @cached_property
    def properties(self):
        match self:
            case LLMProvider.OPENAI:
                props = Asset.PROVIDERS.get(self.value)
                available = litellm_utils.get_available_models(self.value)
                models = set(props.get("models").keys()).union(set(available))
                for model in models:
                    props["models"][model] = props["models"].get(
                        model, dict()
                    )
                    props["models"][model]["context_window"] = (
                        litellm_utils.get_context_window(model)
                    )
                return LLMProviderProperties(**props)
            case LLMProvider.OPENAI_AZURE_SIH:
                return LLMProviderProperties(
                    **Asset.PROVIDERS.get(self.value),
                    models=dict(),
                )


def validate_api_key(
    provider: LLMProvider,
    api_key: str,
) -> bool:
    match provider:
        case _:
            return True
