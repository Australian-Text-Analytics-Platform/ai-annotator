from typing import Callable

import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.techniques.techniques import Technique
from atap_llm_classifier.providers.providers import LLMProvider, validate_api_key
from atap_llm_classifier.views.props import ViewProp, ClassifierConfigProps
from atap_llm_classifier.views import utils

__all__ = [
    "ClassifierConfigView",
]

props: ClassifierConfigProps = ViewProp.CLASSIFIER_CONFIG.properties


class TechniquesSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name=props.technique.selector.name,
            options=[t.value for t in Technique],
        )
        self.desc = pn.pane.Markdown("placeholder", margin=3)
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
        self.desc.object = technique.properties.description
        self.paper_url.object = utils.create_anchor_tag(
            technique.properties.paper_url,
            props.technique.paper_url,
        )

    def disable(self):
        self.selector.disabled = True


class ProviderSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name=props.provider.selector.name,
            options=[llmp.value for llmp in LLMProvider],
        )

        self.desc = pn.widgets.StaticText(value="placeholder", margin=2)
        self.privacy_policy = pn.pane.HTML("placeholder", margin=2)
        self.api_key = pn.widgets.PasswordInput(placeholder="placeholder", width=450)
        self.api_key_msg = pn.pane.Markdown(object=props.provider.api_key.start_message)
        self.layout = pn.Column(
            pn.Row(
                self.selector,
                pn.Column(
                    self.desc,
                    self.privacy_policy,
                ),
            ),
            pn.Row(self.api_key, self.api_key_msg),
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

    @property
    def selected(self) -> LLMProvider:
        return LLMProvider(self.selector.value)

    def _on_select(self, _):
        provider: LLMProvider = self.selected
        self.desc.value = provider.properties.description
        if provider.properties.privacy_policy_url is not None:
            self.privacy_policy.object = utils.create_anchor_tag(
                provider.properties.privacy_policy_url, "Open link to privacy policy🔗"
            )
        else:
            self.privacy_policy.object = (
                f"<span style='color: red; font-weight: bold'>"
                f"{props.provider.no_privacy_url}"
                f"</span>"
            )

        self.api_key.placeholder = (
            f"Enter your {provider.value} API Key here. (Then press 'Enter')"
        )

    def _on_api_key_enter(self, _):
        api_key: str = self.api_key.value
        if validate_api_key(
            api_key=api_key,
            provider=LLMProvider(self.selector.value),
        ):
            self.api_key_msg.object = props.provider.api_key.success_message
            self.api_key.disabled = True
            if self._valid_api_key_callback is not None:
                self._valid_api_key_callback()
        else:
            self.api_key_msg.object = props.provider.api_key.error_message

    def set_valid_api_key_callback(self, callback: Callable):
        self._valid_api_key_callback = callback

    def disable(self):
        self.selector.disabled = True


class ClassifierConfigView(Viewer):
    def __init__(self, **params):
        super(ClassifierConfigView, self).__init__(**params)

        self.techniques = TechniquesSelectorView()
        self.provider = ProviderSelectorView()
        self.layout = pn.Column(
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
