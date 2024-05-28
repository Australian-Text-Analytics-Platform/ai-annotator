from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import ViewPropsWithNameToolTip

__all__ = [
    "PipeModelProps",
]


class LLMInfoProps(BaseModel):
    context_window_prefix: str
    price_per_input_token_prefix: str
    price_per_output_token_prefix: str


class LLMConfigProps(BaseModel):
    selector: ViewPropsWithNameToolTip
    info: LLMInfoProps


class PipeModelProps(BaseModel):
    title: str
    llm: LLMConfigProps
    temperature: ViewPropsWithNameToolTip
    top_p: ViewPropsWithNameToolTip
    seed: ViewPropsWithNameToolTip
