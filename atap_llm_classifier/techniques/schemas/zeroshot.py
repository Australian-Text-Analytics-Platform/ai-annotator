from pydantic import BaseModel

from atap_llm_classifier.techniques.schemas import BasePromptTemplate

__all__ = [
    "ZeroShotPromptTemplate",
    "ZeroShotUserSchema",
    "ZeroShotClass",
]


class ZeroShotUserSchemaTemplates(BaseModel):
    clazz: str


class ZeroShotPromptTemplate(BasePromptTemplate):
    user_schema_templates: ZeroShotUserSchemaTemplates


## User Schema ##


class ZeroShotClass(BaseModel):
    name: str
    description: str


class ZeroShotUserSchema(BaseModel):
    classes: list[ZeroShotClass]
