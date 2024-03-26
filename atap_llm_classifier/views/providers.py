import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.provideres.providers import LLMProvider

_provider_name_to_provider: dict[str, LLMProvider] = {
    llmp.value.name: llmp for llmp in LLMProvider
}


class ProviderSelectorView(Viewer):
    def __init__(self, **params):
        super(ProviderSelectorView, self).__init__(**params)
        self.selector = pn.widgets.Select(
            name="Select an LLM provider:",
            options=[llmp.value.name for llmp in LLMProvider],
        )

        self.desc = pn.widgets.StaticText(value="placeholder", margin=3)
        self.models = pn.widgets.StaticText(value="placeholder", margin=3)
        self.api_key = pn.widgets.PasswordInput(placeholder="placeholder", width=450)
        self.layout = pn.Column(
            pn.Row(
                self.selector,
                pn.Column(
                    self.desc,
                    self.models,
                ),
            ),
            self.api_key,
        )

        self._on_select(None)
        self.selector.param.watch(
            self._on_select,
            "value",
        )

    def __panel__(self) -> Viewable:
        return self.layout

    def _on_select(self, _):
        llmp: LLMProvider = _provider_name_to_provider[self.selector.value]
        self.desc.value = llmp.value.description
        self.models.value = "Available Base Models: " + ", ".join(llmp.value.models)
        self.api_key.placeholder = (
            f"Enter your {llmp.value.name} API Key here. Then press 'Enter'"
        )
