import abc
from typing import Type, Union

import pydantic
from pydantic import BaseModel

from classifier_fastapi.techniques.schemas import BasePromptTemplate

__all__ = [
    "BaseTechnique",
]


class BaseTechnique(metaclass=abc.ABCMeta):
    schema: Type[BaseModel]
    template: BasePromptTemplate
    enable_reasoning: bool = False
    max_reasoning_chars: int = 150

    def __init__(
        self,
        user_schema: Union["BaseTechnique.schema", dict, None],
        enable_reasoning: bool = False,
        max_reasoning_chars: int = 150,
    ):
        try:
            self.user_schema = self.schema.model_validate(user_schema)
        except pydantic.ValidationError as e:
            raise ValueError("Invalid prompt provided for given technique.") from e
        self.enable_reasoning = enable_reasoning
        self.max_reasoning_chars = max_reasoning_chars

    def _get_reasoning_instruction(self) -> str:
        """Generate reasoning instruction if enabled."""
        if not self.enable_reasoning:
            return ""
        return f"\nProvide a brief explanation for your classification (maximum {self.max_reasoning_chars} characters)."

    @abc.abstractmethod
    def make_prompt(self, text: str) -> str:
        raise NotImplementedError()

    @classmethod
    def is_validate_user_schema(
        cls, user_schema: Union["BaseTechnique.schema", dict, None]
    ) -> bool:
        try:
            cls.schema.model_validate(user_schema)
            return True
        except pydantic.ValidationError as _:
            return False
