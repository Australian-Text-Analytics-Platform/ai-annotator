"""Prompting Techniques"""

from enum import Enum
from functools import cached_property

from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.techniques import BaseTechnique
from atap_llm_classifier.techniques.templates import (
    ZeroShotTemplate,
    CoTTemplate,
)

__all__ = [
    "Technique",
    "TechniqueProperties",
]


class TechniqueProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)


class Technique(Enum):
    ZERO_SHOT: str = "zero_shot"
    CHAIN_OF_THOUGHT: str = "chain_of_thought"

    @cached_property
    def properties(self) -> TechniqueProperties:
        return TechniqueProperties(**Asset.TECHNIQUES.get(self.value))

    @cached_property
    def template(self) -> ZeroShotTemplate | CoTTemplate:
        template: dict = Asset.TECH_TEMPLATES.get(self.value)
        match self:
            case Technique.ZERO_SHOT:
                return ZeroShotTemplate(**template)
            case Technique.CHAIN_OF_THOUGHT:
                return CoTTemplate(**template)

    def get_prompt_maker(self, user_schema: BaseModel | dict) -> BaseTechnique:
        match self:
            case Technique.ZERO_SHOT:
                from .zeroshot import ZeroShot

                return ZeroShot(user_schema)
            case Technique.CHAIN_OF_THOUGHT:
                from .cot import ChainOfThought

                return ChainOfThought(user_schema)
