"""Prompting Modifiers"""

import abc
from enum import Enum
from functools import cached_property

from litellm import ModelResponse
from loguru import logger
from pydantic import BaseModel, Field

from classifier_fastapi.assets import Asset
from classifier_fastapi.models import (
    LLMConfig,
)
from classifier_fastapi.techniques import BaseTechnique
from classifier_fastapi.techniques.schemas import LLMoutputModel

__all__ = [
    "Modifier",
    "BaseModifier",
    "NoModifier",
]


class BaseModifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def pre(
        self,
        text: str,
        model: str,
        prompt: str,
        technique: BaseTechnique,
        llm_config: LLMConfig,
    ) -> tuple[str, LLMConfig]:
        raise NotImplementedError()

    @abc.abstractmethod
    def post(
        self,
        response: ModelResponse,
        outputs: list[LLMoutputModel],
        technique: BaseTechnique,
        llm_config: LLMConfig,
        text: str,
        model: str,
    ) -> str:
        raise NotImplementedError()


class NoModifier(BaseModifier):
    def pre(
        self,
        text: str,
        model: str,
        prompt: str,
        technique: BaseTechnique,
        llm_config: LLMConfig,
    ) -> tuple[str, LLMConfig]:
        n = llm_config.n_completions
        if n != 1:
            logger.warning(
                f"Using modifier: {self.__class__.__name__}. Setting number of completions from {n} to 1."
            )
        llm_config.n_completions = 1
        return prompt, llm_config

    def post(
        self,
        response: ModelResponse,
        outputs: list[LLMoutputModel],
        technique: BaseTechnique,
        llm_config: LLMConfig,
        text: str,
        model: str,
    ) -> str:
        if len(outputs) > 1:
            logger.warning(
                f"Modifier: {self.__class__} used. Expecting 1 output but got {len(outputs)}."
            )
        output = outputs[0]
        if output is None:
            logger.warning("Output is None. Using full llm response as classification.")
            return response.choices[0].message.content
        return output.classification


class ModifierInfo(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)
    pre_behaviour: str
    post_behaviour: str


class Modifier(str, Enum):
    NO_MODIFIER: str = "no_modifier"
    SELF_CONSISTENCY: str = "self_consistency"

    @cached_property
    def properties(self) -> ModifierInfo:
        props: dict = Asset.MODIFIERS.get(self.value)
        return ModifierInfo(**props)

    def get_behaviour(self) -> BaseModifier:
        match self:
            case Modifier.NO_MODIFIER:
                return NoModifier()
            case Modifier.SELF_CONSISTENCY:
                from .self_consistency import SelfConsistencyModifier

                return SelfConsistencyModifier()
