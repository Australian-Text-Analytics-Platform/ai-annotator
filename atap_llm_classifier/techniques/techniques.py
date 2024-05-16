"""Prompting Techniques"""

import abc
from abc import ABCMeta
from enum import Enum
from functools import lru_cache, cached_property

from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset

__all__ = ["Technique", "BaseTechnique", "NoTechnique"]


class BaseTechnique(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def make_prompt(self, text: str) -> str:
        raise NotImplementedError()


class NoTechnique(BaseTechnique):
    def make_prompt(self, text: str) -> str:
        return text


class TechniqueProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)


class Technique(Enum):
    CHAIN_OF_THOUGHT: str = "chain_of_thought"

    @cached_property
    def properties(self) -> TechniqueProperties:
        match self:
            case Technique.CHAIN_OF_THOUGHT:
                props: dict = Asset.TECHNIQUES.get(self.value)
                return TechniqueProperties(**props)

    def get_behaviour(self) -> BaseTechnique:
        pass
