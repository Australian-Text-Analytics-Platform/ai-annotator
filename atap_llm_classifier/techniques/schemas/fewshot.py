from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from atap_llm_classifier.techniques.schemas import (
    BasePromptTemplate,
)

__all__ = [
    "FewShotPromptTemplate",
    "FewShotUserSchemaTemplates",
    "FewShotUserSchema",
    "FewShotExample",
    "FewShotClass",
]


## Prompt Template ##


class FewShotUserSchemaTemplates(BaseModel):
    clazz: str
    example: str


class FewShotPromptTemplate(BasePromptTemplate):
    user_schema_templates: FewShotUserSchemaTemplates


## User Schema ##


class FewShotExample(BaseModel):
    query: str
    classification: str


class FewShotClass(BaseModel):
    name: str
    description: str


class FewShotUserSchema(BaseModel):
    classes: list[FewShotClass]
    examples: list[FewShotExample]

    @field_validator("examples", mode="after")
    @classmethod
    def classes_are_defined_for_examples(
        cls, v: list[FewShotExample], info: ValidationInfo
    ):
        uniq_classes = set(map(lambda c: c.name, info.data.get("classes")))
        uniq_ex_classes = set(map(lambda ex: ex.classification, v))
        missing = uniq_ex_classes - uniq_classes
        if len(missing) > 0:
            raise ValueError(
                f"Missing class definitions for examples: {', '.join(list(missing))}"
            )
        return v