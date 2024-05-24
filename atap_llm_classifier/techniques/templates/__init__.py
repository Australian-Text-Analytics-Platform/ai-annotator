from pydantic import BaseModel


class PromptTemplateOutputFormats(BaseModel):
    yaml: str


class BasePromptTemplate(BaseModel):
    structure: str
    output_formats: PromptTemplateOutputFormats


from .cot import *
from .zeroshot import *
