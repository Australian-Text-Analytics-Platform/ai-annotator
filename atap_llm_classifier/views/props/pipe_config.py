from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import SelectorProps

__all__ = [
    "PipeConfigProps",
]


class APIKeyProps(BaseModel):
    placeholder: str
    start_message: str
    success_message: str
    error_message: str


class ProviderConfigProps(BaseModel):
    selector: SelectorProps
    privacy_url: str
    no_privacy_url: str
    api_key: APIKeyProps


class TechniqueConfigProps(BaseModel):
    selector: SelectorProps
    paper_url: str


class ModifierConfigProps(BaseModel):
    selector: SelectorProps
    paper_url: str


class PipeConfigProps(BaseModel):
    technique: TechniqueConfigProps
    modifier: ModifierConfigProps
    provider: ProviderConfigProps
