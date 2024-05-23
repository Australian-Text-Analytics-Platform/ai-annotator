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
        model_props = provider.properties.models
        self.selector = pn.widgets.Select(
            options=sorted(model_props),
            width=250,
            max_width=250,
        )
        model_info = pn.bind(
            lambda selected: pn.pane.Markdown(f"""
        {asset.get("model_info_context_window")} {model_props[selected].context_window}
        {asset.get("model_info_price_per_input_token")} {model_props[selected].input_token_cost}
        {asset.get("model_info_price_per_output_token")} {model_props[selected].output_token_cost}
        
        {model_props[selected].description if model_props[selected].description is not None else ""}
        """),
            self.selector,
        )

        self.layout = pn.Column(
            asset.get("title"),
            self.selector,
            pn.widgets.TooltipIcon(
                value=asset.get("select_model_tooltip"),
                margin=(-33, -450, 20, -170),
            ),
            model_info,
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
                margin=(-43, -450, 30, -170),
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
                margin=(-43, -450, 50, -170),
            ),
        )

        # todo: bind model config to a Paramterised class of LLMConfigs

    @property
    def selected(self) -> str:
        return self.selector.value

    def __panel__(self) -> Viewable:
        return self.layout
