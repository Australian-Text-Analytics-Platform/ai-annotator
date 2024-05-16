"""providers.py"""

from enum import Enum
from functools import lru_cache, cached_property
import re

from pydantic import BaseModel, Field, HttpUrl

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.utils import litellm_ as litellm_utils

__all__ = [
    "LLMProvider",
    "LLMProviderProperties",
]


class LLMModelProperties(BaseModel):
    description: str | None = None
    context_window: int | None = None
    input_token_cost: float
    output_token_cost: float


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


def validate_api_key(
    provider: LLMProvider,
    api_key: str,
) -> bool:
    match provider:
        case _:
            return True
