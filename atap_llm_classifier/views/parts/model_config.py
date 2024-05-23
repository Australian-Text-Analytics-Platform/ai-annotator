import sys

import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.assets import Asset
from atap_llm_classifier.views.props import ViewProp, ModelConfigProps
from atap_llm_classifier.providers.providers import LLMProvider

asset: dict = Asset.VIEWS.get("model_config")
props: ModelConfigProps = ViewProp.MODEL_CONFIG.properties

__all__ = [
    "ModelConfigView",
]


# todo: sensible defaults based on technique
class ModelConfigView(Viewer):
    def __init__(self, **params):
        provider: LLMProvider = params.pop("provider")
        super().__init__(**params)
        model_props = provider.properties.models
        mprops_rx = pn.rx(provider.properties.models)
        self.selector = pn.widgets.Select(
            name=props.llm.selector.name,
            options=sorted(model_props),
            width=250,
            max_width=250,
        )
        mprop_rx = mprops_rx[self.selector.rx()]

        model_info_rx = pn.rx("""
        {mprop.description}

        {props.llm.info.context_window_prefix} {mprop.context_window}
        {props.llm.info.price_per_input_token_prefix} {mprop.input_token_cost}
        {props.llm.info.price_per_output_token_prefix} {mprop.output_token_cost}
        """).format(mprop=mprop_rx, props=props)
        model_info_md = pn.pane.Markdown(model_info_rx)

        self.layout = pn.Column(
            # asset.get("title"),
            props.title,
            self.selector,
            pn.widgets.TooltipIcon(
                # value=asset.get("select_model_tooltip"),
                value=props.llm.selector.tooltip,
                margin=(-33, -450, 20, -170),
            ),
            model_info_md,
            pn.widgets.FloatSlider(
                # name=asset.get("top_p_title"),
                name=props.top_p.name,
                start=0.1,
                end=1.0,
                step=0.1,
                value=0.8,
                tooltips=True,
            ),
            pn.widgets.TooltipIcon(
                # value=asset.get("top_p_tooltip"),
                value=props.top_p.tooltip,
                margin=(-43, -450, 30, -170),
            ),
            pn.widgets.FloatSlider(
                # name=asset.get("temperature_title"),
                name=props.temperature.name,
                start=0.0,
                end=2.0,
                step=0.1,
                value=1.0,
                tooltips=False,
            ),
            pn.widgets.TooltipIcon(
                # value=asset.get("temperature_tooltip"),
                value=props.temperature.tooltip,
                margin=(-43, -450, 50, -170),
            ),
        )

        # todo: bind model config to a Paramterised class of LLMConfigs

    @property
    def selected(self) -> str:
        return self.selector.value

    @property
    def rselected(self):
        return self.selector.param.value

    def __panel__(self) -> Viewable:
        return self.layout
