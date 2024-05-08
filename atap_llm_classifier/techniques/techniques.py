"""Prompting Techniques"""

from enum import Enum
from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset


class TechniqueContext(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)


class Technique(Enum):
    CHAIN_OF_THOUGHT: str = "chain_of_thought"

    def get_context(self):
        match self:
            case Technique.CHAIN_OF_THOUGHT:
                ctx: dict = Asset.TECHNIQUES.get(self.value)
                return TechniqueContext(**ctx)
