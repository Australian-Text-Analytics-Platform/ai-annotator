"""PipelineCosts

Costs are recalculated based on 2 conditions:
1. data is stale (model changed or prompt changed)
2. periodic rerender gate is opened. Set at PERIODIC_RERENDER_INTERVAL_MS.

Due to large corpus, recalculating the cost for all documents is expensive which is why the gate is in place.
Further optimisations likely possible e.g. caching etc.
"""

import panel as pn

from panel.viewable import Viewer

from atap_llm_classifier.views.parts.pipeline_classifications import (
    PipelineClassifications,
)
from atap_llm_classifier.views.parts.pipeline_model import PipelineModelConfigView
from atap_llm_classifier.views.parts.pipeline_prompt import PipelinePrompt

PERIODIC_RERENDER_INTERVAL_MS: int = 5_000


class PipelineCosts(Viewer):
    def __init__(
        self,
        pipe_mconfig: PipelineModelConfigView,
        pipe_prompt: PipelinePrompt,
        pipe_classifications: PipelineClassifications,
        **params,
    ):
        super().__init__(**params)

        self.pipe_mconfig = pipe_mconfig
        self.pipe_prompt = pipe_prompt
        self.pipe_classifications = pipe_classifications

        self._data_stale = pn.rx(lambda *_: True)(
            pipe_mconfig.model_selector, pipe_prompt.user_schema_rx
        )
        self._periodic_rerender_gate = pn.rx(False)
        self.rerender_due = self._data_stale.rx.is_(True).rx.and_(
            self._periodic_rerender_gate.rx.is_(True)
        )
        pn.state.add_periodic_callback(
            self._open_rerender_gate, period=PERIODIC_RERENDER_INTERVAL_MS
        )

        self.total_tokens = pn.rx(self.compute_total_prompt_tokens())
        self.total_cost = pn.rx(self.compute_total_cost())
        pn.bind(self.recalculate, self.rerender_due, watch=True)

        # todo: cost gauge counter to be updated during classification.

        self.layout = pn.Column(
            pn.pane.Str(
                self.total_tokens.rx.pipe("Total number of tokens: {}".format),
                margin=(0, 10),
            ),
            pn.pane.Str(
                self.total_cost.rx.pipe("Minimum $USD: {:.2f}".format),
                margin=(0, 10),
            ),
        )

    def __panel__(self):
        return self.layout

    def recalculate(self, due: bool):
        if due:
            self.total_tokens.rx.value = self.compute_total_prompt_tokens()
            self.total_cost.rx.value = self.compute_total_cost()
            self._close_rerender_gate()

    def compute_total_prompt_tokens(self) -> int | str:
        try:
            return sum(self.pipe_classifications.get_prompt_token_counts())
        except Exception as e:
            return "N/A"

    def compute_total_cost(self) -> float | str:
        mprops = self.pipe_mconfig.mprop_rx.rx.value
        try:
            return mprops.input_token_cost * self.total_tokens.rx.value
        except Exception as e:
            return "N/A"

    def _open_rerender_gate(self):
        self._periodic_rerender_gate.rx.value = True

    def _close_rerender_gate(self):
        self._periodic_rerender_gate.rx.value = False
