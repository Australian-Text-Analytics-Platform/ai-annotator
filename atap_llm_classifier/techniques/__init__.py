import abc
from typing import Type, Union

import pydantic
from pydantic import BaseModel


class BaseTechnique(metaclass=abc.ABCMeta):
    schema: Type[BaseModel]

    def __init__(self, user_schema: Union["BaseTechnique.schema", dict, None]):
        try:
            self.user_schema = self.schema.model_validate(user_schema)
        except pydantic.ValidationError as e:
            raise ValueError("Invalid prompt provided for given technique.") from e

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


from .techniques import *
