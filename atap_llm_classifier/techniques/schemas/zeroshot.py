from pydantic import BaseModel

from atap_llm_classifier.techniques.schemas import BasePromptTemplate

__all__ = [
    "ZeroShotPromptTemplate",
]


class ZeroShotUserSchemaTemplates(BaseModel):
    clazz: str


class ZeroShotPromptTemplate(BasePromptTemplate):
    user_schema_templates: ZeroShotUserSchemaTemplates
