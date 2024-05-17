import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.views.model_config import ModelConfigView
from atap_llm_classifier.providers.providers import LLMProviderProperties, LLMProvider
from atap_corpus import Corpus


class ClassificationWidget(Viewer):
    def __init__(self, provider: LLMProvider, corpus: Corpus, **params):
        super(ClassificationWidget, self).__init__(**params)

        self.mconfig = ModelConfigView(provider=provider)

        # todo: create Accordions after prompt have been uploaded.

        df = corpus.to_dataframe()
        dataset_widget = pn.widgets.DataFrame(
            df,
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
        )

        progress_bar = pn.widgets.Tqdm()

        # note: removed 'dataset' tab - allow classifications to be streamed to the widget.
        # todo: improve prompt tab: include sample prompt for chosen technique. then preview prompt.
        self.tabs = pn.Tabs(
            (
                "Prompt",
                pn.Column(
                    pn.widgets.FileInput(accept=".yaml,.yml"),
                    pn.Accordion(
                        (
                            "placeholder",
                            pn.Column(
                                pn.pane.HTML("Just a placeholder"),
                            ),
                        )
                    ),
                ),
            ),
            (
                "Classifications",
                dataset_widget,
            ),
        )

        self.layout = pn.Column(
            pn.Row(
                pn.Column(
                    self.mconfig,
                    pn.widgets.Button(name="Run classification", button_type="primary"),
                ),
                pn.Spacer(width=20),
                self.tabs,
            ),
            progress_bar,
            sizing_mode="stretch_both",
        )

    def __panel__(self) -> Viewable:
        return self.layout


def create_classifier(
    provider: LLMProvider,
    corpus: Corpus,
) -> ClassificationWidget:
    return ClassificationWidget(provider=provider, corpus=corpus)
