from typing import Callable

import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.views.techniques import TechniquesSelectorView
from atap_llm_classifier.views.providers import ProviderSelectorView


class ClassifierConfigView(Viewer):
    def __init__(self, **params):
        super(ClassifierConfigView, self).__init__(**params)

        self.techniques = TechniquesSelectorView()
        self.provider = ProviderSelectorView()
        self.layout = pn.Column(
            "## LLM Classifier Configuration",
            self.techniques,
            pn.Spacer(height=5),
            self.provider,
        )

    def __panel__(self) -> Viewable:
        return self.layout

    def set_provider_valid_api_callback(self, callback: Callable):
        self.provider.set_valid_api_key_callback(callback)

    def disable(self):
        self.provider.disable()
        self.techniques.disable()
