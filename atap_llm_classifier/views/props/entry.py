from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import SelectorPropsWithDesc

__all__ = [
    "EntryProps",
]


class DatasetProps(BaseModel):
    selector: SelectorPropsWithDesc


class EntryProps(BaseModel):
    title: str
    dataset: DatasetProps
