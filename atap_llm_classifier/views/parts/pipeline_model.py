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


# todo: sensible defaults based on technique
class PipelineModelConfigView(Viewer):
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

        model_info_rx = pn.rx(MODEL_INFO_FORMAT_STR).format(mprop=mprop_rx, props=props)
        model_info_md = pn.pane.Markdown(model_info_rx, width=250)

        self.layout = pn.Column(
            props.title,
            self.selector,
            pn.widgets.TooltipIcon(
                value=props.llm.selector.tooltip,
                margin=(-33, -450, 20, -170),
            ),
            model_info_md,
            pn.widgets.FloatSlider(
                name=props.top_p.name,
                start=0.1,
                end=1.0,
                step=0.1,
                value=0.8,
                tooltips=True,
            ),
            pn.widgets.TooltipIcon(
                value=props.top_p.tooltip,
                margin=(-43, -450, 30, -170),
            ),
            pn.widgets.FloatSlider(
                name=props.temperature.name,
                start=0.0,
                end=2.0,
                step=0.1,
                value=1.0,
                tooltips=False,
            ),
            pn.widgets.TooltipIcon(
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
