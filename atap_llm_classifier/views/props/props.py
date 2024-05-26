import enum
from functools import cached_property
from typing import Type

from pydantic import BaseModel

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.views import props

__all__ = [
    "ViewProp",
]


class ViewProp(enum.Enum):
    ENTRY: str = "entry"
    PIPE_CONFIG: str = "pipe_config"
    PIPE_MODEL: str = "pipe_model"

    @cached_property
    def properties(self):
        cls: Type[BaseModel]
        match self:
            case ViewProp.ENTRY:
                cls = props.EntryProps
            case ViewProp.PIPE_CONFIG:
                cls = props.PipeConfigProps
            case ViewProp.PIPE_MODEL:
                cls = props.PipeModelProps
            case _:
                raise NotImplementedError()

        return cls(**Asset.VIEWS.get(self.value))
