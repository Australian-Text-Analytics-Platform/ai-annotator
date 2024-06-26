"""providers.py"""

from enum import Enum
from functools import cached_property, lru_cache, partial
import re
from typing import Protocol, Any, Callable

import httpx
from litellm import ModelResponse, AuthenticationError
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, field_validator, SecretStr, ConfigDict

from atap_llm_classifier import config
from atap_llm_classifier.assets import Asset
from atap_llm_classifier.utils import litellm_ as litellm_utils
from atap_llm_classifier.utils.prompt import TokenEncoder, get_token_encoder_for_openai
from atap_llm_classifier.utils.utils import make_dummy_request

from . import (
    openai_,
    ollama,
)

__all__ = [
    "LLMProvider",
    "LLMModelProperties",
    "LLMProviderProperties",
    "LLMProviderUserProperties",
    "LLMModelUserProperties",
    "validate_api_key",
]


class LLMProvider(Enum):
    OPENAI: str = "openai"
    OPENAI_AZURE_SIH: str = "openai_azure_sih"
    OLLAMA: str = "ollama"

    @cached_property
    def properties(self) -> "LLMProviderProperties":
        props: dict = Asset.PROVIDERS.get(self.value)
        models: list[dict] = list()

        available = list()
        get_context_window: Callable[[str], int | None] = lambda _: None
        get_price: Callable[[str], tuple[float, float] | tuple[None, None]] = (
            lambda _: (
                None,
                None,
            )
        )
        match self:
            case LLMProvider.OPENAI_AZURE_SIH:
                # todo: SIH azure openai properties are not yet defined.
                raise NotImplementedError(
                    "LLMProvider properties for SIH OpenAI Azure is not yet implemented."
                )
            case LLMProvider.OLLAMA:
                try:
                    available = ollama.get_available_models(
                        endpoint=props.get("endpoint")
                    )
                except httpx.HTTPStatusError:
                    available = list()
                except Exception as e:
                    raise RuntimeError(
                        "Unable to retrieve ollama properties. Schema may have changed."
                    ) from e
            case _:
                available = litellm_utils.get_available_models(self.value)
                get_context_window = litellm_utils.get_context_window
                get_price = litellm_utils.get_price

        model_regex_ptns: list[re.Pattern] = [
            re.compile(ptn) for ptn in props.get("models").keys()
        ]
        for model_key in available:
            values = dict(name=model_key, provider=self)
            values["context_window"] = get_context_window(model_key)
            inp_cost, out_cost = get_price(model_key)
            values["input_token_cost"] = inp_cost
            values["output_token_cost"] = out_cost

            # assign description based on regex pattern of model name found in asset
            values["description"] = ""
            for ptn in model_regex_ptns:
                if ptn.match(model_key) is not None:
                    d = props.get("models")[ptn.pattern]["description"]
                    values["description"] = d
                    break
            models.append(values)
        props["models"] = models
        return LLMProviderProperties(
            provider=self,
            **props,
        )

    @lru_cache
    def get_user_properties(self, api_key: str) -> "LLMProviderUserProperties":
        """Get the llm properties dependent on the user's API Key."""
        if config.mock.enabled:
            raise ValueError(
                "There are no user specific provider properties when in mock mode."
            )

        if not validate_api_key(self, api_key):
            raise ValueError(f"Invalid api key given for provider: {self.value}.")

        props_copy: LLMProviderProperties = self.properties.copy()
        match self:
            case LLMProvider.OPENAI:
                base_models, ft_ftbase_models = openai_.get_available_models_for_user(
                    api_key
                )
                ft_models_prop_list = list()
                for ft, ft_base in ft_ftbase_models:
                    try:
                        ft_model_props = props_copy.get_model_props(ft_base).model_copy(
                            deep=True
                        )
                        ft_model_props.name = ft
                    except ValueError as e:
                        logger.warning(
                            f"Finetuned base model not found. Skipped id={ft}. Err: {e}"
                        )
                        continue
                    ft_models_prop_list.append(ft_model_props)
                props_copy.models = [
                    model for model in props_copy.models if model.name in base_models
                ]
                for ft_model_props in ft_models_prop_list:
                    props_copy.models.append(ft_model_props)
            case LLMProvider.OPENAI_AZURE_SIH:
                raise NotImplementedError(
                    "LLM user models based on SIH's OpenAI on Azure is not yet implemented."
                )
            case LLMProvider.OLLAMA:
                raise NotImplementedError(
                    "LLM user models based on Ollama is not yet implemented."
                )
            case _:
                raise LookupError("Not a valid provider.")

        user_models = [
            LLMModelUserProperties(validated_api_key=api_key, **model.model_dump())
            for model in props_copy.models
        ]
        return LLMProviderUserProperties(
            validated_api_key=api_key,
            models=user_models,
            **props_copy.model_dump(
                exclude={
                    "models",
                }
            ),
        )


class LLMModelProperties(BaseModel):
    name: str
    provider: LLMProvider
    description: str = Field("")
    context_window: int | None = None
    tokeniser_id: str | None = None
    input_token_cost: float | None = None
    output_token_cost: float | None = None

    @field_validator("description", mode="before")
    @classmethod
    def ensure_str(cls, v):
        return "" if v is None else str(v)

    def known_context_window(self) -> bool:
        return self.context_window is not None

    def known_tokeniser(self) -> bool:
        return self.token_encoder is not None

    def known_input_token_cost(self) -> bool:
        return self.input_token_cost is not None

    def known_output_token_cost(self) -> bool:
        return self.output_token_cost is not None

    def count_tokens(self, prompt: str) -> int:
        return len(self._get_token_encoder().encode(prompt))

    @lru_cache
    def _get_token_encoder(self) -> TokenEncoder:
        if not self.known_tokeniser():
            raise LookupError(
                f"Tokeniser is not known for model {self.name} provider={self.provider.value}."
            )
        match self:
            case LLMProvider.OPENAI:
                return get_token_encoder_for_openai(self.name)
            case _:
                raise NotImplementedError(
                    "Tokeniser is known but token encoder behaviour is not yet implemented."
                )


class LLMModelUserProperties(LLMModelProperties):
    validated_api_key: SecretStr

    def __hash__(self):
        return hash((self.name, self.validated_api_key.get_secret_value()))


class LLMProviderProperties(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: LLMProvider

    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    privacy_policy_url: HttpUrl | None = Field(default=None, frozen=True)
    models: list[LLMModelProperties]
    endpoint: HttpUrl | None

    @lru_cache
    def get_model_props(self, model: str) -> LLMModelProperties:
        for model_prop in self.models:
            if model_prop.name == model:
                return model_prop
        raise ValueError(f"{model} does not exist for provider: {self.name}.")

    # override default hash behaviour from BaseModel which does not allow list.
    # needed for lru_cache
    def __hash__(self):
        return hash((self.name, self.provider))


class LLMProviderUserProperties(LLMProviderProperties):
    validated_api_key: SecretStr
    models: list[LLMModelUserProperties]

    def get_model_props(self, model: str) -> LLMModelUserProperties:
        return super().get_model_props(model=model)

    def __hash__(self):
        return hash((self.name, self.validated_api_key.get_secret_value()))


def validate_api_key(
    provider: LLMProvider,
    api_key: str,
) -> bool:
    if config.mock.enabled:
        return True
    try:
        make_dummy_request_to_provider(provider, api_key)
        return True
    except AuthenticationError as e:
        logger.error(f"Failed to validate api key. Err: {e}.")
        return False


### Helpers ###


def make_dummy_request_to_provider(
    provider: LLMProvider,
    api_key: str | None = None,
) -> ModelResponse:
    dummy_model: str
    if provider.properties.api_key_test_model is not None:
        dummy_model = provider.properties.api_key_test_model
    else:
        try:
            dummy_model = next(iter(provider.properties.models)).name
        except StopIteration as _:
            raise RuntimeError(
                f"No available models for llm provider: {provider.value}. Unable to validate api key."
            )
    return make_dummy_request(
        model=dummy_model,
        api_key=api_key,
    )
