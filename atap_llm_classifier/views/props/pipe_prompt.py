from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import ViewPropsWithName

__all__ = [
    "PipePromptProps",
]


class PipePromptLiveEdit(BaseModel):
    classes: ViewPropsWithName
    examples: ViewPropsWithName


class PipePromptPreview(ViewPropsWithName):
    text_placeholder: str


class PipePromptProps(BaseModel):
    prompt_preview: PipePromptPreview
    live_edit: PipePromptLiveEdit
