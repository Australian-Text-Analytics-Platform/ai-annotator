"""Prompting Techniques"""

from enum import Enum
from functools import cached_property
from typing import Type

from pydantic import BaseModel, Field

from classifier_fastapi.assets import Asset
from classifier_fastapi.techniques.base import BaseTechnique
from classifier_fastapi.techniques.schemas import (
    ZeroShotPromptTemplate,
    FewShotPromptTemplate,
    CoTPromptTemplate,
)

__all__ = [
    "Technique",
    "TechniqueInfo",
]


class TechniqueInfo(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)


class Technique(str, Enum):
    ZERO_SHOT: str = "zero_shot"
    FEW_SHOT: str = "few_shot"
    CHAIN_OF_THOUGHT: str = "chain_of_thought"

    @cached_property
    def info(self) -> TechniqueInfo:
        return TechniqueInfo(**Asset.TECHNIQUES.get(self.value))

    @cached_property
    def prompt_template(self) -> ZeroShotPromptTemplate | FewShotPromptTemplate | CoTPromptTemplate:
        template: dict = Asset.PROMPT_TEMPLATES.get(self.value)
        match self:
            case Technique.ZERO_SHOT:
                return ZeroShotPromptTemplate(**template)
            case Technique.FEW_SHOT:
                return FewShotPromptTemplate(**template)
            case Technique.CHAIN_OF_THOUGHT:
                return CoTPromptTemplate(**template)

    @property
    def prompt_maker_cls(self) -> Type[BaseTechnique]:
        match self:
            case Technique.ZERO_SHOT:
                from .zeroshot import ZeroShot

                return ZeroShot
            case Technique.FEW_SHOT:
                from .fewshot import FewShot

                return FewShot
            case Technique.CHAIN_OF_THOUGHT:
                from .cot import ChainOfThought

                return ChainOfThought

    def get_prompt_maker(
        self,
        user_schema: BaseModel | dict,
        enable_reasoning: bool = False,
        max_reasoning_chars: int = 150,
    ) -> BaseTechnique:
        if isinstance(user_schema, dict):
            user_schema = self.prompt_maker_cls.schema(**user_schema)
        return self.prompt_maker_cls(user_schema, enable_reasoning, max_reasoning_chars)
