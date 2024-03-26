from typing import Callable

import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.providers.providers import LLMProvider, validate_api_key

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
        self.api_key_msg = pn.widgets.StaticText(value="👈 Lock it in.")
        self.layout = pn.Column(
            pn.Row(
                self.selector,
                pn.Column(
                    self.desc,
                    self.models,
                ),
            ),
            pn.Row(
                self.api_key,
                self.api_key_msg,
            ),
        )

        self._on_select(None)
        self.selector.param.watch(
            self._on_select,
            "value",
        )
        self.api_key.param.watch(
            self._on_api_key_enter,
            "value",  # value only changes on enter.
        )

        self._valid_api_key_callback = None

    def __panel__(self) -> Viewable:
        return self.layout

    def _on_select(self, _):
        llmp: LLMProvider = _provider_name_to_provider[self.selector.value]
        self.desc.value = llmp.value.description
        self.models.value = "Available Base Models: " + ", ".join(llmp.value.models)
        self.api_key.placeholder = (
            f"Enter your {llmp.value.name} API Key here. (Then press 'Enter')"
        )

    def _on_api_key_enter(self, _):
        api_key: str = self.api_key.value
        if validate_api_key(
            api_key=api_key,
            provider=_provider_name_to_provider[self.selector.value],
        ):
            self.api_key_msg.value = "👍 Valid API Key."
            if self._valid_api_key_callback is not None:
                self._valid_api_key_callback()
        else:
            self.api_key_msg.value = "🙅 Invalid API Key. Please try again."

    def set_valid_api_key_callback(self, callback: Callable):
        self._valid_api_key_callback = callback

    def disable(self):
        self.selector.disabled = True
