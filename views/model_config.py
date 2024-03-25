import panel as pn

from provideres.providers import LLMProviderContext


class ModelConfigView(object):
    def __init__(self, llm_ctx: LLMProviderContext):
        self.selector = pn.widgets.Select(
            options=llm_ctx.models,
        )
        self.mconfig = pn.Column(
            "## Model Configuration",
            self.selector,
            pn.widgets.TooltipIcon(
                value="Select the model you want to use.",
                margin=(-33, -500, 20, -170),
            ),
            pn.widgets.FloatSlider(
                name="Top p", start=0.1, end=1.0, step=0.1, value=0.8, tooltips=True
            ),
            pn.widgets.TooltipIcon(
                value="Increase this ",
                margin=(-43, -40, 30, -170),
            ),
            pn.widgets.FloatSlider(
                name="Temperature",
                start=0.0,
                end=2.0,
                step=0.1,
                value=1.0,
                tooltips=False,
            ),
            pn.widgets.TooltipIcon(
                value="Increase this a more diverse range of output tokens. If 0, it is deterministic.",
                margin=(-43, -120, 50, -170),
            ),
        )
