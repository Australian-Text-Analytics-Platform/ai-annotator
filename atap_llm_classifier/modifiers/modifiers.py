"""Prompting Modifiers"""

from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field


class Order(Enum):
    PRE: str = "applied before classification"
    POST: str = "applied after classification"


class ModifierContext(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)
    order: Order


class Modifier(Enum):
    SELF_CONSISTENCY: str = "self-consistency"

    @lru_cache()
    def get_context(self) -> ModifierContext:
        match self:
            case Modifier.SELF_CONSISTENCY:
                return ModifierContext(
                    name="Chain of Thought with Self Consistency",
                    description="Ask LLM to generate multiple outputs and use the majority vote.",
                    explanation="",
                    paper_url="https://arxiv.org/abs/2203.11171",
                    order=Order.POST,
                )
            case _:
                raise RuntimeError("Not a valid modifier. This should not happen.")
