from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import ViewPropsWithNameDescription

__all__ = [
    "EntryProps",
]


class DatasetProps(BaseModel):
    selector: ViewPropsWithNameDescription


class EntryProps(BaseModel):
    title: str
    dataset: DatasetProps
