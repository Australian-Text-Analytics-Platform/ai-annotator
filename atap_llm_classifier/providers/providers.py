"""providers.py"""

from enum import Enum
from functools import cached_property, lru_cache
import re
from typing import Callable, Self

import httpx
from litellm import ModelResponse, AuthenticationError
from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    AnyUrl,
    field_validator,
    SecretStr,
    ConfigDict,
    computed_field,
)

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
    "validate_api_key",
]


class LLMProvider(str, Enum):
    OPENAI: str = "openai"
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
            case LLMProvider.OLLAMA:
                endpoint = props.get("endpoint")
                if endpoint is not None:
                    try:
                        available = ollama.get_available_models(
                            endpoint=props.get("endpoint")
                        )
                    except httpx.HTTPStatusError:
                        pass
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
            values = dict(name=model_key, provider=self, endpoint=props.get("endpoint"))
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


class LLMModelProperties(BaseModel):
    name: str
    provider: LLMProvider
    description: str = Field("")
    context_window: int | None = None
    input_token_cost: float | None = None
    output_token_cost: float | None = None
    api_key: str | None = None
    endpoint: str | None = None

    @computed_field
    def id(self) -> str:
        return self.name[self.name.rfind("/") + 1 :]

    @field_validator("description", mode="before")
    @classmethod
    def ensure_str(cls, v):
        return "" if v is None else str(v)

    def known_context_window(self) -> bool:
        return self.context_window is not None

    def known_tokeniser(self) -> bool:
        try:
            self._get_token_encoder()
            return True
        except Exception as e:
            return False

    def known_input_token_cost(self) -> bool:
        return self.input_token_cost is not None

    def known_output_token_cost(self) -> bool:
        return self.output_token_cost is not None

    def is_authenticated(self) -> bool:
        return self.api_key is not None

    def count_tokens(self, prompt: str) -> int:
        return len(self._get_token_encoder().encode(prompt))

    @lru_cache
    def _get_token_encoder(self) -> TokenEncoder:
        match self.provider:
            case LLMProvider.OPENAI:
                return get_token_encoder_for_openai(self.name)
            case _:
                raise NotImplementedError(
                    "Tokeniser is known but token encoder behaviour is not yet implemented."
                )

    def __hash__(self):
        hashables = [self.name, self.provider, self.description]
        if self.api_key is not None:
            hashables.append(self.api_key)
        if self.endpoint is not None:
            hashables.append(self.endpoint)
        return hash(tuple(hashables))


class LLMProviderProperties(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: LLMProvider

    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    models: list[LLMModelProperties]
    privacy_policy_url: AnyUrl | None = Field(default=None, frozen=True)
    endpoint: AnyUrl | None = None
    api_key: str | None = None
    api_key_test_model: str | None = None

    @lru_cache
    def get_model_props(self, model: str) -> LLMModelProperties:
        for model_prop in self.models:
            if model_prop.id == model:
                return model_prop
        raise ValueError(f"{model} does not exist for provider: {self.name}.")

    def is_authenticated(self) -> bool:
        return self.api_key is not None

    @lru_cache
    def with_api_key(self, api_key: str) -> Self:
        if not validate_api_key(self.provider, api_key):
            raise ValueError(
                f"Invalid api key given for provider: {self.provider.value}."
            )
        copy_: Self = self.model_copy(deep=True)
        copy_.api_key = api_key
        match self.provider:
            case LLMProvider.OPENAI:
                base_models, ft_ftbase_models = openai_.get_available_models_for_user(
                    api_key
                )
                ft_models = list()
                for ft, ft_base in ft_ftbase_models:
                    try:
                        ft_model = self.get_model_props(ft_base).model_copy(deep=True)
                        ft_model.name = ft
                    except ValueError as e:
                        logger.warning(
                            f"Finetuned model's base model not found. Skipped model={ft}. Err: {e}"
                        )
                        continue
                    ft_models.append(ft_model)
                copy_.models = list(
                    filter(lambda m: m.name in base_models, copy_.models)
                )
                for ft_model in ft_models:
                    copy_.models.append(ft_model)
            case LLMProvider.OLLAMA:
                raise NotImplementedError()
            case _:
                raise LookupError("Not a valid provider.")

        for m in copy_.models:
            m.api_key = copy_.api_key
        return copy_

    def with_endpoint(self, endpoint: str) -> Self:
        # todo: 1. check endpoint is valid
        # todo: 2. replace endpoint
        # todo: 3. adjust models given that endpoint
        # todo: 4. return a copy of self.
        raise NotImplementedError()

    # override default hash behaviour from BaseModel which does not allow list.
    # needed for lru_cache
    def __hash__(self):
        return hash((self.name, self.provider, self.description))


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
