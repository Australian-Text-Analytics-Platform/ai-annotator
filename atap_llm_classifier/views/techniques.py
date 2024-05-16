import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.techniques.techniques import Technique
from .utils import create_anchor_tag


class TechniquesSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name="Select a technique:",
            options=[t.value for t in Technique],
        )
        self.desc = pn.widgets.StaticText(value="placeholder", margin=3)
        self.paper_url = pn.pane.HTML("placeholder", margin=3)
        self.layout = pn.Column(
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
        return self.layout

    def _on_select(self, _):
        technique: Technique = Technique(self.selector.value)
        self.desc.value = technique.properties.description
        self.paper_url.object = create_anchor_tag(
            technique.properties.paper_url, "Open link to associated paper🔗"
        )

    def disable(self):
        self.selector.disabled = True
