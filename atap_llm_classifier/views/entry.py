import panel as pn
from panel.viewable import Viewer, Viewable

from atap_corpus_loader import CorpusLoader
from atap_corpus import Corpus
from pydantic import SecretStr

from atap_llm_classifier.views.parts.pipe_config import PipeConfigView
from atap_llm_classifier.views.parts.pipeline import create_pipeline
from atap_llm_classifier.views.props import ViewProp, EntryProps
from atap_llm_classifier.views import utils, notify

props: EntryProps = ViewProp.ENTRY.properties


class EntryWidget(Viewer):
    def __init__(self, loader: CorpusLoader, **params):
        super(EntryWidget, self).__init__(**params)

        self.loader = loader

        # react to any loader builds and update selectable corpus
        latest_corpus, set_latest_corpus = utils.rx(self.loader.get_latest_corpus())
        self.r_latest_corpus: pn.rx[Corpus] = latest_corpus
        loader.set_build_callback(set_latest_corpus)

        self.r_loader_corpus_names: pn.rx[list[str]] = pn.rx(
            lambda lc: [c.name for c in loader.controller.corpora.items()]
        )(self.r_latest_corpus)

        self.pipe_config = PipeConfigView()
        self.dataset_selector = pn.widgets.Select(
            name=props.dataset.selector.name,
            options=self.r_loader_corpus_names,
        )

        self.enable_pipe_config_on_dataset()
        pn.bind(
            lambda *args: self.enable_pipe_config_on_dataset(),
            self.dataset_selector,
            watch=True,
        )

        self.layout = pn.Column(
            props.title,
            pn.Row(
                self.dataset_selector,
                pn.pane.Markdown(props.dataset.selector.description),
            ),
            self.pipe_config,
        )

        # placeholder for expansion.
        self.layout_init_len = len(self.layout)
        self.pipeline = None

        self.progress_conditions_met = (
            self.dataset_selector.rx()
            .rx.is_not(None)
            .rx.and_(self.pipe_config.technique.selector.rx())
            .rx.is_not(None)
            .rx.and_(self.pipe_config.modifier.selector.rx())
            .rx.is_not(None)
            .rx.and_(self.pipe_config.provider.selector.rx())
            .rx.is_not(None)
            .rx.and_(self.pipe_config.provider.api_key_is_valid_rx)
            .rx.and_(self.pipe_config.provider.privacy_policy_read.rx())
        )
        pn.bind(self.progress_to_pipeline, self.progress_conditions_met, watch=True)

    def __panel__(self) -> Viewable:
        return self.layout

    def enable_pipe_config_on_dataset(self):
        if self.dataset_selector.value is None:
            self.pipe_config.disable()
        else:
            self.pipe_config.enable()

    @notify.catch(raise_err=True)
    def progress_to_pipeline(self, conditions_met: bool):
        if conditions_met:
            self.disable()
            if len(self.layout) <= self.layout_init_len:
                self.pipeline = create_pipeline(
                    corpus=self.loader.get_corpus(
                        corpus_name=self.dataset_selector.value
                    ),
                    provider=self.pipe_config.provider.selected,
                    api_key=self.pipe_config.provider.api_key,
                    technique=self.pipe_config.technique.selected,
                    modifier=self.pipe_config.modifier.selected,
                )
                self.layout.extend(
                    [
                        pn.Spacer(height=10),
                        self.pipeline,
                    ]
                )

    def disable(self):
        self.dataset_selector.disabled = True
        self.pipe_config.disable()

    def enable(self):
        self.dataset_selector.disabled = False
        self.pipe_config.enable()
