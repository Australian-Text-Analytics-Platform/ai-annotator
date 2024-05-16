import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.providers.providers import LLMProvider

asset: dict = Asset.VIEWS.get("model_config")


# todo: sensible defaults based on technique
class ModelConfigView(Viewer):
    def __init__(self, **params):
        provider: LLMProvider = params.pop("provider")
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            options=sorted(provider.properties.models),
        )
        self.layout = pn.Column(
            asset.get("title"),
            self.selector,
            pn.widgets.TooltipIcon(
                value=asset.get("select_model_tooltip"),
                margin=(-33, -500, 20, -170),
            ),
            pn.widgets.FloatSlider(
                name=asset.get("top_p_title"),
                start=0.1,
                end=1.0,
                step=0.1,
                value=0.8,
                tooltips=True,
            ),
            pn.widgets.TooltipIcon(
                value=asset.get("top_p_tooltip"),
                margin=(-43, -40, 30, -170),
            ),
            pn.widgets.FloatSlider(
                name=asset.get("temperature_title"),
                start=0.0,
                end=2.0,
                step=0.1,
                value=1.0,
                tooltips=False,
            ),
            pn.widgets.TooltipIcon(
                value=asset.get("temperature_tooltip"),
                margin=(-43, -120, 50, -170),
            ),
        )

    def __panel__(self) -> Viewable:
        return self.layout
