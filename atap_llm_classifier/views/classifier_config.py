import panel as pn
from panel.viewable import Viewer, Viewable

from .techniques import TechniquesSelectorView
from .providers import ProviderSelectorView


class ClassifierConfigView(Viewer):
    def __init__(self, **params):
        super(ClassifierConfigView, self).__init__(**params)
        self.layout = pn.Column(
            "## Classifier Configuration",
            TechniquesSelectorView(),
            pn.Spacer(height=5),
            ProviderSelectorView(),
        )

    def __panel__(self) -> Viewable:
        return self.layout
