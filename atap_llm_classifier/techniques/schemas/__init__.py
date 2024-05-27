import abc
from functools import cached_property

from pydantic import BaseModel, Field, ConfigDict


class LLMoutputModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    classification: str


class BasePromptTemplate(BaseModel, metaclass=abc.ABCMeta):
    structure: str
    output_classification_key: str = Field("classification", frozen=True)
    additional_output_keys: list[str] = list()

    user_schema_templates: BaseModel

    @cached_property
    def output_keys(self) -> list[str]:
        return [self.output_classification_key] + self.additional_output_keys


from .cot import *
from .zeroshot import *
