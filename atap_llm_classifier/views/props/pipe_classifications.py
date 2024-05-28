from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import (
    ViewPropsWithNameWidth,
    ViewPropsWithName,
)

__all__ = [
    "PipeClassificationsProps",
]


class PipeClassClassifyOne(BaseModel):
    button: ViewPropsWithNameWidth
    doc_index_selector: ViewPropsWithName
    doc_index_preview: ViewPropsWithName


class PipeClassClassifyAll(BaseModel):
    button: ViewPropsWithNameWidth


class PipeClassStatusMsgs(BaseModel):
    pending: str


class PipeClassClassify(BaseModel):
    one: PipeClassClassifyOne
    all: PipeClassClassifyAll
    status_messages: PipeClassStatusMsgs


class PipeClassCorpusCols(BaseModel):
    document: ViewPropsWithName
    classification: ViewPropsWithName


class PipeClassCorpus(BaseModel):
    columns: PipeClassCorpusCols


class PipeClassificationsProps(BaseModel):
    classify: PipeClassClassify
    corpus: PipeClassCorpus
