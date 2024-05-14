"""Prompting Modifiers"""

import abc
from enum import Enum
from functools import lru_cache, singledispatchmethod
from typing import Any

from litellm import ModelResponse
from loguru import logger
from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.models import (
    LLMModelConfig,
    LiteLLMMessage,
    LiteLLMArgs,
    LiteLLMRole,
)

__all__ = [
    "Modifier",
    "BaseModifier",
    "NoModifier",
]


class BaseModifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def pre(
        self,
        prompt: str,
        llm_config: LLMModelConfig,
    ) -> tuple[str, LLMModelConfig]:
        return prompt, llm_config

    @abc.abstractmethod
    def post(
        self,
        response: ModelResponse,
    ) -> str:
        raise NotImplementedError()


class NoModifier(BaseModifier):
    def pre(
        self, prompt: str, llm_config: LLMModelConfig
    ) -> tuple[str, LLMModelConfig]:
        n = llm_config.n_completions
        if n != 1:
            logger.warning(
                f"Using modifier: {self.__class__.__name__}. Setting number of completions from {n} to 1."
            )
        llm_config.n_completions = 1
        return prompt, llm_config

    def post(self, response: ModelResponse) -> str:
        return response.choices[0].message.content


class Order(Enum):
    PRE: str = "applied before classification"
    POST: str = "applied after classification"


class ModifierProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)
    order: Order


class Modifier(Enum):
    SELF_CONSISTENCY: str = "self_consistency"

    @lru_cache()
    def get_properties(self) -> ModifierProperties:
        match self:
            case Modifier.SELF_CONSISTENCY:
                ctx: dict = Asset.MODIFIERS.get(self.value)
                ctx["order"] = Order.POST
                return ModifierProperties(**ctx)
