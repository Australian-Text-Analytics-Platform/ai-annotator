from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import ViewPropsWithName

__all__ = [
    "PipePromptProps",
]


class PipePromptLiveEditProps(BaseModel):
    classes: ViewPropsWithName
    examples: ViewPropsWithName


class PipePromptProps(BaseModel):
    prompt_preview: ViewPropsWithName
    live_edit: PipePromptLiveEditProps
