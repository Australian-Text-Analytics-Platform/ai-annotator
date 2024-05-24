from pydantic import BaseModel

from atap_llm_classifier.techniques.templates import BasePromptTemplate

__all__ = [
    "CoTTemplate",
    "CoTUserSchemaTemplates",
]


class CoTUserSchemaTemplates(BaseModel):
    clazz: str
    example: str


class CoTTemplate(BasePromptTemplate):
    user_schema_templates: CoTUserSchemaTemplates
