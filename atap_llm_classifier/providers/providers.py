"""classifier.py

Provide the core functionalities of the generic LLM classifier.

1. select provider
2. provide the list of models that I can select from for this provider.
3. given the model, prompt and list of texts, run the batch classification. (stream=False version)
"""

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from atap_llm_classifier.providers.open_ai import (
    list_models as open_ai_list_models,
    validate_api_key as open_ai_validate_api_key,
)
from atap_llm_classifier.providers.azure_open_ai import (
    list_models as azure_open_ai_list_models,
    validate_api_key as azure_open_ai_validate_api_key,
)

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
    models: list[str] = Field(frozen=True)
    key: str = Field(frozen=True)


def create_llm_context(name: str, description: str, key: str) -> LLMProviderContext:
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
    OPENAI: LLMProviderContext = create_llm_context(
        name="OpenAI",
        description="OpenAI the company.",
        key="open_ai",
    )
    AZURE_OPENAI: LLMProviderContext = create_llm_context(
        name="Azure OpenAI (SIH)",
        description="Same as OpenAI but hosted in SIH's Azure Cloud.",
        key="azure_open_ai",
    )
    # note: extend here for more providers.


def list_available_llm_providers() -> list[str]:
    """List all available LLM Providers's name."""
    return list(map(lambda llmp: llmp.value.name, (llmp for llmp in LLMProvider)))


def validate_api_key(api_key: str, provider: LLMProvider) -> bool:
    return _validate_api_key_registry[provider.value.key](api_key)
