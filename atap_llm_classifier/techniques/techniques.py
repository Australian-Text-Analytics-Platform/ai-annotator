"""Prompting Techniques"""

from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset

__all__ = [
    "Technique",
]


class TechniqueProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)


class Technique(Enum):
    CHAIN_OF_THOUGHT: str = "chain_of_thought"

    @lru_cache
    def get_properties(self):
        match self:
            case Technique.CHAIN_OF_THOUGHT:
                ctx: dict = Asset.TECHNIQUES.get(self.value)
                return TechniqueProperties(**ctx)
