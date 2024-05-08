"""classifier.py

Provide the core functionalities of the generic LLM classifier.

1. select provider
2. provide the list of models that I can select from for this provider.
3. given the model, prompt and list of texts, run the batch classification. (stream=False version)
"""

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, HttpUrl, field_validator, computed_field

from atap_llm_classifier.providers.open_ai import (
    list_models as open_ai_list_models,
    validate_api_key as open_ai_validate_api_key,
)
from atap_llm_classifier.providers.azure_open_ai import (
    list_models as azure_open_ai_list_models,
    validate_api_key as azure_open_ai_validate_api_key,
)

__all__ = [
    "LLMProvider",
]

_list_models_registry: dict[str, Callable[..., list[str]]] = {
    "open_ai": open_ai_list_models,
    "azure_open_ai": azure_open_ai_list_models,
}

_validate_api_key_registry: dict[str, Callable[..., bool]] = {
    "open_ai": open_ai_validate_api_key,
    "azure_open_ai": azure_open_ai_validate_api_key,
}


class LLMProviderContext(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    key: str = Field(frozen=True)
    privacy_policy_url: HttpUrl | None = Field(default=None, frozen=True)

    @field_validator("key", mode="after")
    @classmethod
    def key_must_be_in_registry(cls, v: str):
        if v not in _list_models_registry.keys():
            raise KeyError(f"{v} is not valid key for a provider.")
        return v

    @computed_field
    def models(self) -> list[str]:
        return _list_models_registry[self.key]()


def create_llm_context(
    name: str,
    description: str,
    key: str,
) -> LLMProviderContext:
    try:
        models: list[str] = _list_models_registry[key]()
    except KeyError as ke:
        raise KeyError(f"{key} is not valid key for a provider.") from ke

    return LLMProviderContext(
        name=name,
        description=description,
        models=models,
        key=key,
    )


class LLMProvider(Enum):
    OPENAI: LLMProviderContext = LLMProviderContext(
        name="OpenAI",
        description="OpenAI the company.",
        key="open_ai",
        privacy_policy_url="https://openai.com/policies/privacy-policy",
    )
    AZURE_OPENAI: LLMProviderContext = LLMProviderContext(
        name="Azure OpenAI (SIH)",
        description="Same as OpenAI but hosted in SIH's Azure Cloud.",
        key="azure_open_ai",
        privacy_policy_url=None,
    )
    # note: extend here for more providers.


def list_available_llm_providers() -> list[str]:
    """List all available LLM Providers's name."""
    return list(map(lambda llmp: llmp.value.name, (llmp for llmp in LLMProvider)))


def validate_api_key(api_key: str, provider: LLMProvider) -> bool:
    return _validate_api_key_registry[provider.value.key](api_key)
