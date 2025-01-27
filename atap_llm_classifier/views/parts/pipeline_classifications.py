import asyncio

import panel as pn
from panel.viewable import Viewer, Viewable

from atap_corpus import Corpus

from atap_llm_classifier import core, pipeline
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.views import notify
from atap_llm_classifier.views.props import ViewProp, PipeClassificationsProps
from atap_llm_classifier.views.parts.pipeline_model import PipelineModelConfigView
from atap_llm_classifier.views.parts.pipeline_prompt import PipelinePrompt

props: PipeClassificationsProps = ViewProp.PIPE_CLASSIFICATIONS.properties


class PipelineClassifications(Viewer):
    def __init__(
        self,
        corpus: Corpus,
        technique: Technique,
        modifier: Modifier,
        pipe_mconfig: PipelineModelConfigView,
        pipe_prompt: PipelinePrompt,
        **params,
    ):
        super().__init__(**params)
        self.corpus: Corpus = corpus
        self.technique: Technique = technique
        self.modifier: Modifier = modifier
        self.pipe_mconfig: PipelineModelConfigView = pipe_mconfig
        self.pipe_prompt: PipelinePrompt = pipe_prompt

        self.corpus_rx = pn.rx(corpus)

        # todo: this means edits aren't exactly changing the corpus - left for now.
        #   fixed df_widget to have editing disabled - disabled=True.
        self.df = corpus.docs().to_frame(name=props.corpus.columns.document.name)
        self.df[props.corpus.columns.classification.name] = [
            "" for _ in range(len(self.corpus))
        ]

        self.df_widget = pn.widgets.DataFrame(
            self.df,
            autosize_mode="none",
            widths={
                props.corpus.columns.document.name: 600,
                props.corpus.columns.classification.name: 250,
                props.corpus.columns.num_tokens.name: 60,
            },
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
        )

        self.classify_one_btn = pn.widgets.Button(
            name=props.classify.one.button.name,
            width=props.classify.one.button.width,
            margin=(15, 5),
        )
        self.one_idx_inp = pn.widgets.IntInput(
            name="#",
            value=0,
            start=0,
            end=len(self.corpus) - 1,
            width=50,
        )
        self.one_idx_rx = self.one_idx_inp.param.value.rx()
        self.one_doc_rx: pn.rx[str] = self.corpus_rx[self.one_idx_rx].rx.pipe(str)
        self.one_doc_md: pn.pane.Markdown = pn.pane.Markdown(
            pn.rx("{}\n{}".format)(
                props.classify.one.doc_index_preview.name,
                self.one_doc_rx,
            ),
            margin=(0, 20),
        )
        self.classify_one_btn.on_click(self.on_click_classify_one_and_patch_df)

        self.classify_all_btn = pn.widgets.Button(
            name=props.classify.all.button.name,
            width=props.classify.all.button.width,
            margin=(0, self.classify_one_btn.margin[1]),
        )
        self.classify_all_btn.on_click(self.on_click_classify_all_and_patch_df)
        self.all_progress_bar = pn.widgets.Tqdm(
            max=len(corpus),
            width=600,
        )

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

        self.last_batch_results: pipeline.BatchResults | None = None

    def __panel__(self) -> Viewable:
        return self.layout

    def lock_model_config(self):
        if not self.pipe_mconfig.disabled_rx.rx.value:
            self.pipe_mconfig.disable()
            notify.PipelineClassification.MODEL_CONFIG_LOCKED_ON_CLASSIFY.info()

    @notify.catch(raise_err=False)
    async def on_click_classify_one_and_patch_df(self, _):
        try:
            self.classify_one_btn.disabled = True
            self.lock_model_config()

            idx: int = self.one_idx_rx.rx.value
            text: str = self.one_doc_rx.rx.value
            self._df_patch_pending(indices=idx)

            res: core.ClassificationResult = await core.a_classify(
                text=text,
                model=self.pipe_mconfig.user_model.name,
                api_key=self.pipe_mconfig.provider_user_props.api_key,
                llm_config=self.pipe_mconfig.llm_config,
                technique=self.technique.get_prompt_maker(
                    self.pipe_prompt.user_schema_rx.rx.value
                ),
                modifier=self.modifier.get_behaviour(),
            )
            self._df_patch_classification(idx, res.classification)
        except Exception as e:
            raise e
        finally:
            self.classify_one_btn.disabled = False

    @notify.catch(raise_err=False)
    async def on_click_classify_all_and_patch_df(self, _):
        try:
            self.classify_all_btn.disabled = True
            self.lock_model_config()

            self.all_progress_bar.value = 0

            self._df_patch_pending(indices=list(range(len(self.corpus))))
            await asyncio.sleep(
                2
            )  # note: wait until all pending... is rendered. (not ideal)

            def _on_result_cb(result: pipeline.BatchResult):
                classification: str = result.classification_result.classification
                doc_idx: int = result.doc_idx
                self._df_patch_classification(
                    idx=doc_idx, classification=classification
                )
                self.all_progress_bar.value += 1

            batch_results: pipeline.BatchResults = await pipeline.a_batch(
                corpus=self.corpus,
                model_props=self.pipe_mconfig.user_model,
                llm_config=self.pipe_mconfig.llm_config,
                technique=self.technique,
                user_schema=self.pipe_prompt.user_schema,
                modifier=self.modifier,
                on_result_callback=_on_result_cb,
            )
            self.all_progress_bar.value = self.all_progress_bar.max
            self.last_batch_results = batch_results
        except Exception as e:
            raise e
        finally:
            self.classify_all_btn.disabled = False

    def _df_patch_pending(self, indices: int | list[int]):
        if isinstance(indices, int):
            indices = [indices]
        patches = [(idx, props.classify.status_messages.pending) for idx in indices]
        self.df_widget.patch({props.corpus.columns.classification.name: patches})

    def _df_patch_classification(self, idx: int, classification: str):
        patch = (idx, classification)
        self.df_widget.patch({props.corpus.columns.classification.name: [patch]})

    def df_update_doc_input_tokens(self, num_tokens: list[int | str]):
        patches = list(
            zip(
                range(len(self.df_widget.value)),
                num_tokens,
            )
        )
        self.df_widget.patch({props.corpus.columns.num_tokens.name: patches})

    def get_prompt_token_counts(self) -> list[int | str]:
        def _try_token_count_or_na(text: str) -> int | str:
            if not self.pipe_mconfig.user_model.known_tokeniser():
                return props.corpus.columns.num_tokens.err_value
            else:
                return self.pipe_mconfig.user_model.count_tokens(text)

        prompt_maker = self.technique.get_prompt_maker(self.pipe_prompt.user_schema)
        return list(
            map(
                _try_token_count_or_na,
                map(
                    prompt_maker.make_prompt,
                    self.df[props.corpus.columns.document.name],
                ),
            )
        )
