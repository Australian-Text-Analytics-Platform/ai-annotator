"""Prompting Techniques"""

from enum import Enum
from pydantic import BaseModel, Field


class TechniqueContext(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)


class Technique(Enum):
    CHAIN_OF_THOUGHT: TechniqueContext = TechniqueContext(
        name="Chain of Thought",
        description="",
        explanation="",
        paper_url="https://arxiv.org/abs/2201.11903",
    )
    SELF_CONSISTENCY: TechniqueContext = TechniqueContext(
        name="Chain of Thought with Self Consistency",
        description="",
        explanation="",
        paper_url="https://arxiv.org/abs/2203.11171",
    )
