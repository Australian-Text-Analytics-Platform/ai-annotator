import panel as pn
from panel.viewable import Viewer, Viewable
from pydantic import BaseModel

from atap_corpus_loader import CorpusLoader
from atap_corpus import Corpus
from atap_llm_classifier.views.classifier_config import ClassifierConfigView
from atap_llm_classifier.views.classifier import create_classifier
from atap_llm_classifier.views.props import ViewProp, EntryProps
from atap_llm_classifier.views import utils

props: EntryProps = ViewProp.ENTRY.properties


class EntryWidget(Viewer):
    def __init__(self, loader: CorpusLoader, **params):
        super(EntryWidget, self).__init__(**params)

        self.loader = loader

        # react to any loader builds
        latest_corpus, set_latest_corpus = utils.rx(self.loader.get_latest_corpus())
        self.r_latest_corpus: pn.rx[Corpus] = latest_corpus
        loader.set_build_callback(set_latest_corpus)

        self.r_loader_corpus_names: pn.rx[list[str]] = pn.rx(
            lambda lc: [c.name for c in loader.controller.corpora.items()]
        )(self.r_latest_corpus)

        self.select_dataset = pn.widgets.Select(
            name=props.dataset.selector.name,
            options=self.r_loader_corpus_names,
        )

        self.classifier_config = ClassifierConfigView()
        self.classifier_config.set_provider_valid_api_callback(
            self.classifier_valid_api_key_callback
        )
        self.layout = pn.Column(
            props.title,
            pn.Row(
                self.select_dataset,
                pn.pane.Markdown(props.dataset.selector.description),
            ),
            self.classifier_config,
        )
        self.layout_init_len = len(self.layout)
        self.classifier = None

    def classifier_valid_api_key_callback(self):
        self.select_dataset.disabled = True
        self.classifier_config.disable()
        if len(self.layout) <= self.layout_init_len:
            self.classifier = create_classifier(
                self.classifier_config.provider.selected,
                self.loader.get_corpus(corpus_name=self.select_dataset.value),
            )
            self.layout.extend(
                [
                    pn.Spacer(height=10),
                    self.classifier,
                ]
            )

    def __panel__(self) -> Viewable:
        return self.layout
