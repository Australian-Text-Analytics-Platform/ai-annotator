from pydantic import BaseModel

from atap_llm_classifier.techniques.schemas import BasePromptTemplate

__all__ = [
    "ZeroShotPromptTemplate",
    "ZeroShotSchema",
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


class ZeroShotSchema(BaseModel):
    classes: list[ZeroShotClass]
