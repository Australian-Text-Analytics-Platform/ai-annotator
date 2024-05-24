from pydantic import BaseModel

from atap_llm_classifier.techniques.templates import BasePromptTemplate

__all__ = [
    "ZeroShotTemplate",
]


class ZeroShotUserSchemaTemplates(BaseModel):
    clazz: str


class ZeroShotTemplate(BasePromptTemplate):
    user_schema_templates: ZeroShotUserSchemaTemplates
