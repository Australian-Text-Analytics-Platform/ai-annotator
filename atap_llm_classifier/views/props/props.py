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
    CLASSIFIER_CONFIG: str = "classifier_config"

    @cached_property
    def properties(self):
        cls: Type[BaseModel]
        match self:
            case ViewProp.ENTRY:
                cls = props.EntryProps
            case ViewProp.CLASSIFIER_CONFIG:
                cls = props.ClassifierConfigProps
            case _:
                raise NotImplementedError()

        return cls(**Asset.VIEWS.get(self.value))
