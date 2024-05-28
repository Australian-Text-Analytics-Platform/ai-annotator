import panel as pn
from panel.viewable import Viewer, Viewable

from atap_corpus import Corpus
from pydantic import SecretStr

from atap_llm_classifier import core
from atap_llm_classifier.models import LLMConfig
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.views import notify
from atap_llm_classifier.views.parts.pipeline_model import PipelineModelConfigView
from atap_llm_classifier.views.parts.pipeline_prompt import PipelinePrompt


class PipelineClassifications(Viewer):
    def __init__(
        self,
        corpus: Corpus,
        api_key: SecretStr,
        technique: Technique,
        modifier: Modifier,
        pipe_mconfig: PipelineModelConfigView,
        pipe_prompt: PipelinePrompt,
        **params,
    ):
        super().__init__(**params)
        self.corpus: Corpus = corpus
        self.api_key: SecretStr = api_key
        self.technique: Technique = technique
        self.modifier: Modifier = modifier
        self.pipe_mconfig: PipelineModelConfigView = pipe_mconfig
        self.pipe_prompt: PipelinePrompt = pipe_prompt

        self.corpus_rx = pn.rx(corpus)

        df = corpus.docs().to_frame(name="document")
        df["classification"] = ["" for _ in range(len(self.corpus))]

        self.df_widget = pn.widgets.DataFrame(
            df,
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
        )

        self.classify_one_btn = pn.widgets.Button(
            name="Classify One",
            width=150,
            margin=(15, 5),
        )
        self.one_idx_inp = pn.widgets.IntInput(
            name="#",
            value=0,
            start=0,
            end=len(self.corpus),
            width=50,
        )
        self.one_idx_rx = self.one_idx_inp.param.value.rx()
        self.one_doc_rx: pn.rx[str] = self.corpus_rx[self.one_idx_rx].rx.pipe(str)
        self.one_doc_md: pn.pane.Markdown = pn.pane.Markdown(
            pn.rx("Document preview:\n{}".format)(self.one_doc_rx), margin=(0, 20)
        )
        self.classify_one_btn.on_click(self.on_click_classify_one_and_patch_df)

        self.classify_all_btn = pn.widgets.Button(
            name="Classify All",
            width=self.classify_one_btn.width,
            margin=(0, self.classify_one_btn.margin[1]),
        )
        self.all_progress_bar = pn.widgets.Tqdm()
        self.layout = pn.Column(
            self.df_widget,
            pn.Row(
                self.classify_one_btn,
                self.one_idx_inp,
                self.one_doc_md,
            ),
            pn.Row(
                self.classify_all_btn,
                self.all_progress_bar,
            ),
        )

    def __panel__(self) -> Viewable:
        return self.layout

    def lock_model_config(self):
        if not self.pipe_mconfig.disabled_rx.rx.value:
            self.pipe_mconfig.disable()
            notify.PipelineClassification.MODEL_CONFIG_LOCKED_ON_CLASSIFY.info()

    @notify.catch(raise_err=False)
    async def on_click_classify_one_and_patch_df(self, _):
        self.lock_model_config()

        idx: int = self.one_idx_rx.rx.value
        text: str = self.one_doc_rx.rx.value
        self.df_widget.patch({"classification": [(idx, "pending...")]})

        res: core.ClassificationResult = await core.a_classify(
            text=text,
            model=self.pipe_mconfig.model,
            api_key=self.api_key.get_secret_value(),
            llm_config=LLMConfig(
                temperature=self.pipe_mconfig.temperature,
                top_p=self.pipe_mconfig.top_p,
                seed=self.pipe_mconfig.seed,
            ),
            technique=self.technique.get_prompt_maker(
                self.pipe_prompt.user_schema_rx.rx.value
            ),
            modifier=self.modifier.get_behaviour(),
        )
        self.df_widget.patch({"classification": [(idx, res.classification)]})

    @notify.catch(raise_err=False)
    async def classify_all(self):
        self.lock_model_config()
