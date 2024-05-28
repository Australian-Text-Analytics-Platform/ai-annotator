import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.providers.providers import LLMProvider
from atap_llm_classifier.views.props import ViewProp, PipeModelProps

props: PipeModelProps = ViewProp.PIPE_MODEL.properties

__all__ = [
    "PipelineModelConfigView",
]

MODEL_INFO_FORMAT_STR: str = """
{mprop.description}

{props.llm.info.context_window_prefix} {mprop.context_window}
{props.llm.info.price_per_input_token_prefix} {mprop.input_token_cost}
{props.llm.info.price_per_output_token_prefix} {mprop.output_token_cost}
"""

COMPONENT_WIDTH: int = 300
COMP_TOOLTIP_MARGIN: tuple[int, int, int, int] = (0, 0, 0, 5)


# todo: sensible defaults based on technique
class PipelineModelConfigView(Viewer):
    def __init__(self, **params):
        provider: LLMProvider = params.pop("provider")
        super().__init__(**params)
        model_props = provider.properties.models
        mprops_rx = pn.rx(provider.properties.models)

        self.model_selector = pn.widgets.Select(
            name=props.llm.selector.name,
            options=sorted(model_props),
            width=COMPONENT_WIDTH,
        )
        self.model_row = pn.Row(
            self.model_selector,
            pn.widgets.TooltipIcon(
                value=props.llm.selector.tooltip,
                margin=COMP_TOOLTIP_MARGIN,
            ),
        )

        # model information (reacts to selector's value)
        self.mprop_rx = mprops_rx[self.model_selector.rx()]
        self.model_info_rx = pn.rx(MODEL_INFO_FORMAT_STR).format(
            mprop=self.mprop_rx, props=props
        )
        self.model_info_md = pn.pane.Markdown(self.model_info_rx, width=250)

        self.top_p_slider = pn.widgets.FloatSlider(
            name=props.top_p.name,
            start=0.1,
            end=1.0,
            step=0.1,
            value=0.8,
            tooltips=True,
            width=self.model_selector.width,
        )
        self.top_p_row = pn.Row(
            self.top_p_slider,
            pn.widgets.TooltipIcon(
                value=props.top_p.tooltip,
                margin=COMP_TOOLTIP_MARGIN,
            ),
            sizing_mode="stretch_width",
        )

        self.temp_slider = pn.widgets.FloatSlider(
            name=props.temperature.name,
            start=0.0,
            end=2.0,
            step=0.1,
            value=1.0,
            tooltips=True,
            width=self.model_selector.width,
        )
        self.temp_row = pn.Row(
            self.temp_slider,
            pn.widgets.TooltipIcon(
                value=props.temperature.tooltip,
                margin=COMP_TOOLTIP_MARGIN,
            ),
            width=self.model_selector.width,
        )

        self.disabled = pn.rx(False)
        self.enable_btn = pn.widgets.Button(
            name="Unlock",
            visible=self.disabled.rx.is_(True),
        )
        self.enable_btn.on_click(lambda *args: self.enable())

        self.layout = pn.Column(
            props.title,
            self.model_row,
            self.model_info_md,
            self.top_p_row,
            self.temp_row,
            self.enable_btn,
            width=350,
        )

    def __panel__(self) -> Viewable:
        return self.layout

    @property
    def model(self) -> str:
        return self.model_selector.value

    @property
    def top_p(self) -> float:
        return self.top_p_slider.value

    @property
    def temperature(self) -> float:
        return self.temp_slider.value

    def disable(self):
        self.model_selector.disabled = True
        self.top_p_slider.disabled = True
        self.temp_slider.disabled = True
        self.disabled.rx.value = True

    def enable(self):
        self.model_selector.disabled = False
        self.top_p_slider.disabled = False
        self.temp_slider.disabled = False
        self.disabled.rx.value = False
