import panel as pn
from panel.viewable import Viewer, Viewable
from pydantic import SecretStr

from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.views.parts.pipe_cost import PipelineCosts
from atap_llm_classifier.views.parts.pipeline_classifications import (
    PipelineClassifications,
)
from atap_llm_classifier.views.parts.pipeline_prompt import PipelinePrompt
from atap_llm_classifier.views.parts.pipeline_model import PipelineModelConfigView
from atap_llm_classifier.providers.providers import (
    LLMProvider,
    LLMProviderProperties,
)
from atap_corpus import Corpus

__all__ = [
    "PipelineWidget",
]


class PipelineWidget(Viewer):
    def __init__(
        self,
        corpus: Corpus,
        provider: LLMProvider,
        api_key: SecretStr,
        technique: Technique,
        modifier: Modifier,
        **params,
    ):
        super(PipelineWidget, self).__init__(**params)
        self.corpus: Corpus = corpus
        self.provider: LLMProvider = provider
        self.provider_props: LLMProviderProperties = self.provider.properties.with_api_key(api_key.get_secret_value())
        self.technique: Technique = technique
        self.modifier: Modifier = modifier

        self.pipe_prompt = PipelinePrompt(
            technique=self.technique,
        )
        self.pipe_mconfig = PipelineModelConfigView(
            provider_user_props=self.provider_props,
        )
        self.pipe_classifs = PipelineClassifications(
            corpus=self.corpus,
            technique=self.technique,
            modifier=self.modifier,
            pipe_mconfig=self.pipe_mconfig,
            pipe_prompt=self.pipe_prompt,
        )
        self.pipe_costs = PipelineCosts(
            pipe_mconfig=self.pipe_mconfig,
            pipe_prompt=self.pipe_prompt,
            pipe_classifications=self.pipe_classifs,
        )

        self.tabs = pn.Tabs(
            (
                "Prompt",
                self.pipe_prompt,
            ),
            (
                "Classifications",
                self.pipe_classifs,
            ),
        )

        self.layout = pn.Column(
            pn.Row(
                pn.Column(
                    self.pipe_mconfig,
                    pn.Spacer(height=20),
                    self.pipe_costs,
                ),
                pn.Spacer(width=20),
                self.tabs,
            ),
            sizing_mode="stretch_both",
        )

    def __panel__(self) -> Viewable:
        return self.layout


def create_pipeline(
    corpus: Corpus,
    provider: LLMProvider,
    api_key: SecretStr,
    technique: Technique,
    modifier: Modifier,
) -> PipelineWidget:
    return PipelineWidget(
        corpus=corpus,
        provider=provider,
        api_key=api_key,
        technique=technique,
        modifier=modifier,
    )
