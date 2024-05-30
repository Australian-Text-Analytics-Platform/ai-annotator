"""providers.py"""

from enum import Enum
from functools import cached_property, lru_cache
import re

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, field_validator, SecretStr
import litellm

from atap_llm_classifier.settings import get_settings
from atap_llm_classifier.assets import Asset
from atap_llm_classifier.utils import litellm_ as litellm_utils
from atap_llm_classifier.models import LiteLLMMessage, LiteLLMRole, LiteLLMArgs
from atap_llm_classifier.ratelimiters import RateLimit

__all__ = [
    "LLMProvider",
    "LLMProviderProperties",
]


class LLMModelProperties(BaseModel):
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
    models: dict[str, LLMModelProperties]
    default_rate_limit: RateLimit | None = None


class LLMProviderUserProperties(LLMProviderProperties):
    api_key: SecretStr


class LLMProvider(Enum):
    OPENAI: str = "openai"
    OPENAI_AZURE_SIH: str = "openai_azure_sih"

    @cached_property
    def properties(self):
        match self:
            case LLMProvider.OPENAI_AZURE_SIH:
                return LLMProviderProperties(
                    **Asset.PROVIDERS.get(self.value),
                    models=dict(),
                )
            case _:
                props = Asset.PROVIDERS.get(self.value)
                available = litellm_utils.get_available_models(self.value)
                model_regex_ptns: list[re.Pattern] = [
                    re.compile(ptn) for ptn in props.get("models").keys()
                ]
                models = dict()
                for model_key in available:
                    values = dict()
                    values["description"] = None
                    values["context_window"] = litellm_utils.get_context_window(
                        model_key
                    )
                    inp_cost, out_cost = litellm_utils.get_price(model_key)
                    values["input_token_cost"] = inp_cost
                    values["output_token_cost"] = out_cost
                    for ptn in model_regex_ptns:
                        if ptn.match(model_key) is not None:
                            d = props.get("models")[ptn.pattern]["description"]
                            values["description"] = d
                            break
                    models[model_key] = values
                props["models"] = models
                return LLMProviderProperties(**props)

    @lru_cache
    def get_user_properties(self, api_key: str) -> LLMProviderUserProperties:
        # todo: validate api key
        props: LLMProviderProperties = self.properties
        # todo: pop model if not available from this api key.
        return


def validate_api_key(
    provider: LLMProvider,
    api_key: str,
) -> bool:
    if get_settings().USE_MOCK:
        return True

    model_to_try: str
    match provider:
        case LLMProvider.OPENAI:
            model_to_try = "gpt-3.5-turbo"
        case _:
            try:
                model_to_try = next(iter(provider.properties.models.keys()))
            except StopIteration as e:
                raise RuntimeError(
                    f"No available models for llm provider: {provider.value}. Unable to validate api key."
                )
    try:
        msg = LiteLLMMessage(content="Say Yes.", role=LiteLLMRole.USER)
        litellm.completion(
            **LiteLLMArgs(
                model=model_to_try,
                messages=[msg],
                temperature=0,
                top_p=1.0,
                n=1,
                api_key=api_key,
            ).to_kwargs(),
            max_tokens=10,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to validate api key. Err: {e}.")
        return False
