"""Prompting Techniques"""

import abc
from typing import Type, Union
from enum import Enum
from functools import cached_property

import pydantic
from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.techniques.cot import ChainOfThought

__all__ = ["Technique", "BaseTechnique", "NoTechnique"]


class BaseTechnique(metaclass=abc.ABCMeta):
    prompt_schema: Type[BaseModel]

    def __init__(self, prompt: Union["BaseTechnique.prompt_schema", dict, None]):
        try:
            self.prompt = self.prompt_schema.model_validate(prompt)
        except pydantic.ValidationError as e:
            raise ValueError("Invalid prompt provided for given technique.") from e

    @abc.abstractmethod
    def make_prompt(self, text: str) -> str:
        raise NotImplementedError()


class NoTechnique(BaseTechnique):
    def __init__(self):
        super().__init__(prompt=None)

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

    def get_behaviour(self, prompt: BaseModel | dict) -> BaseTechnique:
        match self:
            case Technique.CHAIN_OF_THOUGHT:
                return ChainOfThought(prompt)
