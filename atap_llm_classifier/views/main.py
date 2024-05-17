import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.views.classifier_config import ClassifierConfigView
from atap_llm_classifier.views.classifier import create_classifier
from atap_corpus_loader import CorpusLoader
from atap_corpus import Corpus


class MainWidget(Viewer):
    def __init__(self, corpus: Corpus, **params):
        super(MainWidget, self).__init__(**params)
        self.corpus = corpus

        self.classifier_config = ClassifierConfigView()
        self.classifier_config.set_provider_valid_api_callback(
            self.classifier_valid_api_key_callback
        )
        self.layout = pn.Column(
            self.classifier_config,
        )

    def classifier_valid_api_key_callback(self):
        self.classifier_config.disable()
        if len(self.layout) <= 1:
            self.layout.append(pn.Spacer(height=10))
            self.layout.append(
                create_classifier(
                    self.classifier_config.provider.selected,
                    self.corpus,
                )
            )

    def __panel__(self) -> Viewable:
        return self.layout
