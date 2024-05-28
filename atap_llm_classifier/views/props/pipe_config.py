from pydantic import BaseModel

from atap_llm_classifier.views.props.shared import ViewPropsWithName

__all__ = [
    "PipeConfigProps",
]


class APIKeyProps(BaseModel):
    placeholder: str
    start_message: str
    success_message: str
    error_message: str


class ProviderConfigProps(BaseModel):
    selector: ViewPropsWithName
    privacy_url: str
    no_privacy_url: str
    api_key: APIKeyProps


class TechniqueConfigProps(BaseModel):
    selector: ViewPropsWithName
    paper_url: str


class ModifierConfigProps(BaseModel):
    selector: ViewPropsWithName
    paper_url: str


class PipeConfigProps(BaseModel):
    technique: TechniqueConfigProps
    modifier: ModifierConfigProps
    provider: ProviderConfigProps
