from litellm import ModelResponse

from .modifiers import BaseModifier
from ..models import LLMConfig

__all__ = [
    "SelfConsistencyModifier",
]


class SelfConsistencyModifier(BaseModifier):
    def pre(self, prompt: str, llm_config: LLMConfig) -> tuple[str, LLMConfig]:
        return "", llm_config

    def post(self, response: ModelResponse) -> str:
        return ""
