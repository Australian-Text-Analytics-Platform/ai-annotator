import panel as pn
from panel.viewable import Viewer, Viewable

from atap_corpus import Corpus


class PipelineClassifications(Viewer):
    def __init__(self, corpus: Corpus, **params):
        super().__init__(**params)
        self.corpus = corpus
        self.corpus_rx = pn.rx(corpus)

        df = corpus.docs().to_frame(name="document")

        self.df_widget = pn.widgets.DataFrame(
            df,
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
        )

        classify_one_btn = pn.widgets.Button(
            name="Classify One",
            width=120,
            margin=(15, 5),
        )
        one_idx_inp = pn.widgets.IntInput(
            name="doc index",
            value=0,
            start=0,
            end=len(self.corpus),
            width=80,
        )
        one_idx_doc = pn.pane.Markdown(self.corpus_rx[one_idx_inp.rx()], margin=(10, 20))
        classify_all_btn = pn.widgets.Button(
            name="Classify All",
            width=classify_one_btn.width,
            margin=(0, classify_one_btn.margin[1]),
        )
        all_progress_bar = pn.widgets.Tqdm()
        self.layout = pn.Column(
            self.df_widget,
            pn.Row(
                classify_one_btn,
                one_idx_inp,
                one_idx_doc,
            ),
            pn.Row(
                classify_all_btn,
                all_progress_bar,
            ),
        )

    def __panel__(self) -> Viewable:
        return self.layout
