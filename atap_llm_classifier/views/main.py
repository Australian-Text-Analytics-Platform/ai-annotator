import panel as pn
from panel.viewable import Viewer, Viewable

from .classifier_config import ClassifierConfigView
from .classification import create_classification_widget


class MainWidget(Viewer):
    def __init__(self, **params):
        super(MainWidget, self).__init__(**params)

        self.classifier = ClassifierConfigView()
        self.classifier.set_provider_valid_api_callback(
            self.classifier_valid_api_key_callback
        )
        self.layout = pn.Column(
            self.classifier,
        )

    def classifier_valid_api_key_callback(self):
        self.classifier.disable()
        if len(self.layout) <= 1:
            from .providers import _provider_name_to_provider

            provider = _provider_name_to_provider[
                self.classifier.provider.selector.value
            ]
            self.layout.append(pn.Spacer(height=10))
            self.layout.append(create_classification_widget(provider.value))

    def __panel__(self) -> Viewable:
        return self.layout
