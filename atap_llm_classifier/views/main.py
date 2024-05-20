from typing import Any, Callable

import panel as pn
from panel.viewable import Viewer, Viewable

from atap_corpus_loader import CorpusLoader
from atap_llm_classifier.views.classifier_config import ClassifierConfigView
from atap_llm_classifier.views.classifier import create_classifier
from atap_corpus import Corpus


def rx(obj: Any) -> tuple[pn.rx, Callable]:
    robj = pn.rx(obj)

    def set_obj(new: Any):
        robj.rx.value = new

    return robj, set_obj


class MainWidget(Viewer):
    def __init__(self, loader: CorpusLoader, **params):
        super(MainWidget, self).__init__(**params)

        # todo: make corpus reactive
        corpus, set_corpus = rx(None)
        loader.set_build_callback(set_corpus)
        self.corpus = corpus

        self.classifier_config = ClassifierConfigView()
        self.classifier_config.set_provider_valid_api_callback(
            self.classifier_valid_api_key_callback
        )
        self.layout = pn.Column(
            self.classifier_config,
        )
        self.classifier = None

    def classifier_valid_api_key_callback(self):
        self.classifier_config.disable()
        if len(self.layout) <= 1:
            self.classifier = create_classifier(
                self.classifier_config.provider.selected,
                self.corpus,
            )
            self.layout.extend(
                [
                    pn.Spacer(height=10),
                    self.classifier,
                ]
            )

    def __panel__(self) -> Viewable:
        return self.layout
