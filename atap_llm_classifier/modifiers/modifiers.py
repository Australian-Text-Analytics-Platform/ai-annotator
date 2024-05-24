"""Prompting Modifiers"""

import abc
from enum import Enum
from functools import lru_cache, singledispatchmethod, cached_property
from typing import Any

from litellm import ModelResponse
from loguru import logger
from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.models import (
    LLMConfig,
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
        llm_config: LLMConfig,
    ) -> tuple[str, LLMConfig]:
        raise NotImplementedError()

    @abc.abstractmethod
    def post(
        self,
        response: ModelResponse,
    ) -> str:
        raise NotImplementedError()


class NoModifier(BaseModifier):
    def pre(self, prompt: str, llm_config: LLMConfig) -> tuple[str, LLMConfig]:
        n = llm_config.n_completions
        if n != 1:
            logger.warning(
                f"Using modifier: {self.__class__.__name__}. Setting number of completions from {n} to 1."
            )
        llm_config.n_completions = 1
        return prompt, llm_config

    def post(self, response: ModelResponse) -> str:
        return response.choices[0].message.content


class ModifierProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)
    pre_behaviour: str
    post_behaviour: str


class Modifier(Enum):
    NO_MODIFIER: str = "no_modifier"
    SELF_CONSISTENCY: str = "self_consistency"

    @cached_property
    def properties(self) -> ModifierProperties:
        props: dict = Asset.MODIFIERS.get(self.value)
        return ModifierProperties(**props)

    def get_behaviour(self) -> BaseModifier:
        match self:
            case Modifier.NO_MODIFIER:
                return NoModifier()
            case Modifier.SELF_CONSISTENCY:
                from .self_consistency import SelfConsistencyModifier

                return SelfConsistencyModifier()
