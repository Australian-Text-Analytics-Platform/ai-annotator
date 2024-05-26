import abc

from pydantic import BaseModel


class BasePromptTemplate(BaseModel, metaclass=abc.ABCMeta):
    structure: str
    output_keys: list[str]


from .cot import *
from .zeroshot import *
