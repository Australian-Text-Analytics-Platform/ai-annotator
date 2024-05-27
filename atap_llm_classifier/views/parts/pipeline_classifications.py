import panel as pn
from panel.viewable import Viewer, Viewable

from atap_corpus import Corpus


class PipelineClassifications(Viewer):
    def __init__(self, corpus: Corpus, **params):
        super().__init__(**params)
        self.corpus = corpus

        df = corpus.docs().to_frame(name="document")

        self.df_widget = pn.widgets.DataFrame(
            df,
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
        )

        # todo: classify one
        btn = pn.widgets.Button(name="Run")
        self.layout = pn.Column(
            pn.Row(btn),
            self.df_widget,
        )

    def __panel__(self) -> Viewable:
        return self.layout
