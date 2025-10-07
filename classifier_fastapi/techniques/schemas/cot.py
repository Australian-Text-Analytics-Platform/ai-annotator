import pydantic
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing import Optional

from classifier_fastapi.techniques.schemas import (
    BasePromptTemplate,
)

__all__ = [
    "CoTPromptTemplate",
    "CoTUserSchemaTemplates",
    "CoTUserSchema",
    "CoTExample",
    "CoTClass",
]


## Prompt Template ##


class CoTUserSchemaTemplates(BaseModel):
    clazz: str
    example: str


class CoTPromptTemplate(BasePromptTemplate):
    user_schema_templates: CoTUserSchemaTemplates


## User Schema ##


class CoTExample(BaseModel):
    query: str
    classification: str
    reason: Optional[str] = None


class CoTClass(BaseModel):
    name: str
    description: str


class CoTUserSchema(BaseModel):
    classes: list[CoTClass]
    examples: list[CoTExample]

    @field_validator("examples", mode="after")
    @classmethod
    def classes_are_defined_for_examples(
        cls, v: list[CoTExample], info: ValidationInfo
    ):
        uniq_classes = set(map(lambda c: c.name, info.data.get("classes")))
        uniq_ex_classes = set(map(lambda ex: ex.classification, v))
        missing = uniq_ex_classes - uniq_classes
        if len(missing) > 0:
            raise pydantic.ValidationError(
                f"Missing class but in example. {', '.join(list(missing))}"
            )
        return v
