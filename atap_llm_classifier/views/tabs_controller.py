import panel as pn
from panel.viewable import Viewer, Viewable
import pandas as pd

from atap_llm_classifier.views.model_config import ModelConfigView
from atap_llm_classifier.providers.providers import LLMProviderProperties


class ClassificationWidget(Viewer):
    def __init__(self, **params):
        llm_ctx = params.pop("llm_ctx")
        super(ClassificationWidget, self).__init__(**params)

        df = pd.DataFrame()

        instructions = """
        You are an expert in philosophy, spefically in the domain of biases.
        You are trying to identify biases that deterministically or categorically link genes to "traits" or "phenotypes", including obese, obesity, overweight, diabetic, diabetes, heavy, fat, fatness; and also behaviours such as eating, overeating, hunger, hungry, craving, fat storage, weight gain, gaining weight, weight loss, losing weight, exercise, physical activity, burning calories.\
        Sentence must be about genes causing a trait, not about a trait affecting a gene.
        When genes "do not", "don't", or "didn't" cause or determine or lead to a trait, then no bias. 
        """

        html_str = """
        <h3>Instructions</h3>
        {instructions}
        """

        mconfig = ModelConfigView(llm_ctx=llm_ctx)

        clz_collapsible = pn.Accordion(
            (
                "class",
                pn.Column(
                    pn.pane.HTML(html_str.format(instructions=instructions).lstrip()),
                    pn.pane.DataFrame(
                        df, index=False, height=200, sizing_mode="stretch_width"
                    ),
                ),
            )
        )

        dataset_df = pd.DataFrame(
            [f"query {i}" for i in range(100)], columns=["sentence"]
        )

        dataset_widget = pn.widgets.DataFrame(
            dataset_df,
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
        )

        tqdm = pn.widgets.Tqdm()

        controller = pn.Row(
            pn.Column(
                mconfig,
                pn.widgets.Button(name="Run classification", button_type="primary"),
            ),
            pn.Spacer(width=20),
            pn.Tabs(
                (
                    "Prompt",
                    pn.Column(
                        pn.widgets.FileInput(accept=".yaml,.yml"), clz_collapsible
                    ),
                ),
                (
                    "Dataset",
                    pn.Column(
                        pn.widgets.FileInput(accept=".csv,.xlsx"), dataset_widget
                    ),
                ),
                (
                    "Classifications",
                    pn.pane.DataFrame(),
                ),
            ),
        )
        self.layout = pn.Column(
            controller,
            tqdm,
            sizing_mode="stretch_both",
        )

    def __panel__(self) -> Viewable:
        return self.layout


def create_classification_widget(llm_ctx: LLMProviderProperties) -> ClassificationWidget:
    return ClassificationWidget(llm_ctx=llm_ctx)
