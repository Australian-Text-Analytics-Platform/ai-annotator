import asyncio
from functools import partial

import panel as pn
from loguru import logger
from panel.viewable import Viewer, Viewable

from atap_corpus import Corpus

from atap_llm_classifier import core
from atap_llm_classifier.models import LLMConfig
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.views.parts.pipeline_model import PipelineModelConfigView
from atap_llm_classifier.views.parts.pipeline_prompt import PipelinePrompt

import os


class PipelineClassifications(Viewer):
    def __init__(
        self,
        corpus: Corpus,
        pipe_mconfig: PipelineModelConfigView,
        pipe_prompt: PipelinePrompt,
        **params,
    ):
        super().__init__(**params)
        self.corpus = corpus
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
            name="doc index",
            value=0,
            start=0,
            end=len(self.corpus),
            width=80,
        )
        self.one_idx_doc = pn.pane.Markdown(
            self.corpus_rx[self.one_idx_inp.rx()], margin=(10, 20)
        )

        self.classify_one_btn.on_click(lambda _: self.classify_one())

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
                self.one_idx_doc,
            ),
            pn.Row(
                self.classify_all_btn,
                self.all_progress_bar,
            ),
        )

    def __panel__(self) -> Viewable:
        return self.layout

    async def classify_one(self):
        idx = self.one_idx_inp.value
        self.df_widget.patch({"classification": [(idx, "pending...")]})

        res: core.Result = await core.a_classify(
            text="hello",
            model="gpt-3.5-turbo",
            api_key="",
            llm_config=LLMConfig(seed=42),
            technique=self.pipe_prompt.get_prompt_maker(),
            modifier=Modifier.NO_MODIFIER.get_behaviour(),
        )
        self.df_widget.patch({"classification": [(idx, res.classification)]})

    async def classify_all(self):
        pass
