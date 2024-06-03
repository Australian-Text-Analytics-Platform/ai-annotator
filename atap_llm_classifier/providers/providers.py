"""providers.py"""

from enum import Enum
from functools import cached_property, lru_cache
import re

import tiktoken
from litellm import ModelResponse, AuthenticationError
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, field_validator, SecretStr

from atap_llm_classifier import config
from atap_llm_classifier.assets import Asset
from atap_llm_classifier.utils import litellm_ as litellm_utils
from atap_llm_classifier.utils.utils import make_dummy_request

__all__ = [
    "LLMProvider",
    "LLMModelProperties",
    "LLMProviderProperties",
    "LLMProviderUserProperties",
    "LLMUserModelProperties",
    "validate_api_key",
    "exceeds_context_window",
]


class LLMProvider(Enum):
    OPENAI: str = "openai"
    OPENAI_AZURE_SIH: str = "openai_azure_sih"

    @cached_property
    def properties(self) -> "LLMProviderProperties":
        props = Asset.PROVIDERS.get(self.value)
        match self:
            case LLMProvider.OPENAI_AZURE_SIH:
                # todo: SIH azure openai properties are not yet defined.
                props["models"] = list()
            case _:
                available = litellm_utils.get_available_models(self.value)
                model_regex_ptns: list[re.Pattern] = [
                    re.compile(ptn) for ptn in props.get("models").keys()
                ]
                models: list[dict] = list()
                for model_key in available:
                    values = dict(name=model_key, provider=self)
                    values["description"] = None
                    values["context_window"] = litellm_utils.get_context_window(
                        model_key
                    )
                    inp_cost, out_cost = litellm_utils.get_price(model_key)
                    values["input_token_cost"] = inp_cost
                    values["output_token_cost"] = out_cost
                    # assign description based on regex pattern of model name found in asset
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
        props_copy: LLMProviderProperties = self.properties.copy()
        if not validate_api_key(self, api_key):
            raise ValueError(f"Invalid api key given for provider: {self.value}.")
        if not config.mock:
            match self:
                case LLMProvider.OPENAI:
                    user_models = get_available_openai_models_for_user(api_key)
                    props_copy.models = [
                        model
                        for model in props_copy.models
                        if model.name in user_models
                    ]
                    # amend finetuned models
                    for finetuned in [
                        user_model
                        for user_model in user_models
                        if user_model.startswith("ft")
                    ]:
                        model = finetuned.split(":")[1]
                        try:
                            ft_model_props = props_copy.get_model_props(
                                model
                            ).model_copy(deep=True)
                        except ValueError as e:
                            logger.warning(
                                f"Finetuned base model not found. Skipped id={finetuned}. Err: {e}"
                            )
                            continue
                        ft_model_props.name = finetuned
                        props_copy.models.append(ft_model_props)
        user_models = [
            LLMUserModelProperties(validated_api_key=api_key, **model.model_dump())
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
    input_token_cost: float
    output_token_cost: float

    @field_validator("description", mode="before")
    @classmethod
    def ensure_str(cls, v):
        return "" if v is None else str(v)


class LLMProviderProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    privacy_policy_url: HttpUrl | None = Field(default=None, frozen=True)
    models: list[LLMModelProperties]
    provider: LLMProvider

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


class LLMUserModelProperties(LLMModelProperties):
    validated_api_key: SecretStr

    def __hash__(self):
        return hash((self.name, self.validated_api_key.get_secret_value()))


class LLMProviderUserProperties(LLMProviderProperties):
    validated_api_key: SecretStr
    models: list[LLMUserModelProperties]

    def get_model_props(self, model: str) -> LLMUserModelProperties:
        return super().get_model_props(model=model)

    def __hash__(self):
        return hash((self.name, self.validated_api_key.get_secret_value()))


def exceeds_context_window(
    model_props: LLMModelProperties,
    prompt: str,
) -> bool | None:
    pass  # todo


def validate_api_key(
    provider: LLMProvider,
    api_key: str,
) -> bool:
    if config.mock:
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
    api_key: str,
) -> ModelResponse:
    dummy_model: str
    match provider:
        case LLMProvider.OPENAI:
            dummy_model = "gpt-3.5-turbo"
        case _:
            try:
                dummy_model = next(iter(provider.properties.models)).name
            except StopIteration as _:
                raise RuntimeError(
                    f"No available models for llm provider: {provider.value}. Unable to validate api key."
                )
    return make_dummy_request(model=dummy_model, api_key=api_key)


def get_available_openai_models_for_user(
    api_key: str,
) -> set[str]:
    from openai import OpenAI
    from openai.pagination import SyncPage

    client = OpenAI(api_key=api_key)
    paginator: SyncPage = client.models.list()
    avail_user_models: set[str] = {
        model.id for page in paginator.iter_pages() for model in page.data
    }
    return avail_user_models
