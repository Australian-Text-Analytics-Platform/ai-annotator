import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.providers.providers import LLMProviderProperties, LLMProvider


class ModelConfigView(Viewer):
    def __init__(self, **params):
        provider: LLMProvider = params.pop("provider")
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            options=sorted(provider.properties.models),
        )
        self.layout = pn.Column(
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

    def __panel__(self) -> Viewable:
        return self.layout

    # todo: the above values should be based on the technique chosen.
