import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.techniques.techniques import Technique

_techq_name_to_technique: dict[str, Technique] = {t.value.name: t for t in Technique}


def create_anchor_tag(link: str) -> str:
    return f"<a href={link} target='_blank'>Open link to associated paper🔗</a>"


class TechniquesSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name="Select a technique:",
            options=[t.value.name for t in Technique],
        )
        self.desc = pn.widgets.StaticText(value="placeholder")
        self.paper_url = pn.pane.HTML("placeholder")
        self.classifier_config = pn.Column(
            "## Classifier Configuration",
            pn.Row(
                self.selector,
                pn.Column(
                    self.desc,
                    self.paper_url,
                ),
            ),
        )
        self._on_select(None)
        self.selector.param.watch(
            self._on_select,
            "value",
        )

    def __panel__(self) -> Viewable:
        return self.classifier_config

    def _on_select(self, _):
        techq: Technique = _techq_name_to_technique[self.selector.value]
        self.desc.value = techq.value.description
        self.paper_url.object = create_anchor_tag(techq.value.paper_url)
