"""Prompting Techniques"""

from enum import Enum
from pydantic import BaseModel, Field


class TechniqueContext(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)


class Techniques(Enum):
    CHAIN_OF_THOUGHT: TechniqueContext = TechniqueContext(
        name="Chain of Thought",
        description="",
        explanation="",
    )
    SELF_CONSISTENCY: TechniqueContext = TechniqueContext(
        name="Chain of Thought with Self Consistency",
        description="",
        explanation="",
    )
