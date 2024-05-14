"""Prompting Modifiers"""

from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field

from atap_llm_classifier.assets import Asset

__all__ = [
    "Modifier",
]


class Order(Enum):
    PRE: str = "applied before classification"
    POST: str = "applied after classification"


class ModifierProperties(BaseModel):
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    explanation: str = Field(frozen=True)
    paper_url: str = Field(default="", frozen=True)
    order: Order


class Modifier(Enum):
    SELF_CONSISTENCY: str = "self_consistency"

    @lru_cache()
    def get_properties(self) -> ModifierProperties:
        match self:
            case Modifier.SELF_CONSISTENCY:
                ctx: dict = Asset.MODIFIERS.get(self.value)
                ctx["order"] = Order.POST
                return ModifierProperties(**ctx)
