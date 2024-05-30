import inspect
from typing import Callable

import panel as pn
from panel.viewable import Viewer, Viewable
from pydantic import SecretStr

from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.providers.providers import LLMProvider, validate_api_key
from atap_llm_classifier.techniques.techniques import Technique
from atap_llm_classifier.views import utils, notify
from atap_llm_classifier.views.props import ViewProp, PipeConfigProps

__all__ = [
    "PipeConfigView",
]

props: PipeConfigProps = ViewProp.PIPE_CONFIG.properties


class TechniqueSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name=props.technique.selector.name,
            options=[t.value for t in Technique],
        )
        self.desc = pn.pane.Markdown("placeholder", margin=3, height=25)
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

    @property
    def selected(self) -> Technique:
        return Technique(self.selector.value)

    def __panel__(self) -> Viewable:
        return self.layout

    def _on_select(self, _):
        technique: Technique = self.selected
        self.desc.object = technique.info.description
        self.paper_url.object = utils.create_anchor_tag(
            technique.info.paper_url,
            props.technique.paper_url,
        )

    def disable(self):
        self.selector.disabled = True

    def enable(self):
        self.selector.disabled = False


class ModifierSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name=props.modifier.selector.name,
            options=[m.value for m in Modifier],
        )
        self.desc = pn.pane.Markdown("placeholder", margin=3, height=25)
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

    @property
    def selected(self) -> Modifier:
        return Modifier(self.selector.value)

    def __panel__(self) -> Viewable:
        return self.layout

    def _on_select(self, _):
        mod: Modifier = Modifier(self.selector.value)
        self.desc.object = mod.properties.description
        self.paper_url.object = utils.create_anchor_tag(
            mod.properties.paper_url,
            props.modifier.paper_url,
        )

    def disable(self):
        self.selector.disabled = True

    def enable(self):
        self.selector.disabled = False


class ProviderSelectorView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self.selector = pn.widgets.Select(
            name=props.provider.selector.name,
            options=[llmp.value for llmp in LLMProvider],
        )

        self.desc = pn.widgets.StaticText(value="placeholder", margin=3)
        self.privacy_policy = pn.pane.HTML("placeholder", margin=3)
        self.api_key_inp = pn.widgets.PasswordInput(
            placeholder="placeholder", width=450
        )
        self.api_key_msg = pn.pane.Markdown(object=props.provider.api_key.start_message)
        self.api_key_is_valid_rx = pn.rx(False)

        self._on_select(None)
        self.selector.param.watch(
            self._on_select,
            "value",
        )
        self.api_key_inp.param.watch(
            self._on_api_key_enter,
            "value",  # value only changes on enter.
        )
        self._valid_api_key_callback = None

        self.layout = pn.Column(
            pn.Row(
                self.selector,
                pn.Column(
                    self.desc,
                    self.privacy_policy,
                ),
            ),
            pn.Row(self.api_key_inp, self.api_key_msg),
        )

    def __panel__(self) -> Viewable:
        return self.layout

    @property
    def api_key(self) -> SecretStr:
        return SecretStr(self.api_key_inp.value)

    @property
    def selected(self) -> LLMProvider:
        return LLMProvider(self.selector.value)

    def _on_select(self, _):
        provider: LLMProvider = self.selected
        self.desc.value = provider.properties.description
        if provider.properties.privacy_policy_url is not None:
            self.privacy_policy.object = utils.create_anchor_tag(
                provider.properties.privacy_policy_url,
                props.provider.privacy_url,
            )
        else:
            self.privacy_policy.object = (
                f"<span style='color: red; font-weight: bold'>"
                f"{props.provider.no_privacy_url}"
                f"</span>"
            )

        self.api_key_inp.placeholder = props.provider.api_key.placeholder

    @notify.catch(raise_err=False)
    def _on_api_key_enter(self, _):
        api_key: SecretStr = SecretStr(self.api_key_inp.value)
        if validate_api_key(
            api_key=api_key.get_secret_value(),
            provider=LLMProvider(self.selector.value),
        ):
            self.api_key_msg.object = props.provider.api_key.success_message
            self.api_key_is_valid_rx.rx.value = True
            self.disable()
            if self._valid_api_key_callback is not None:
                self._valid_api_key_callback(api_key=api_key)
        else:
            self.api_key_msg.object = props.provider.api_key.error_message

    def set_valid_api_key_callback(self, callback: Callable):
        num_args_in_callback = len(inspect.signature(callback).parameters)
        if num_args_in_callback < 1:
            raise ValueError(
                "Valid API key callback must accept at least one parameter which is the api_key as SecretStr."
            )
        self._valid_api_key_callback = callback

    def disable(self):
        self.selector.disabled = True
        self.api_key_inp.disabled = True

    def enable(self):
        self.selector.disabled = False
        self.api_key_inp.disabled = False


class PipeConfigView(Viewer):
    def __init__(self, **params):
        super().__init__(**params)

        self.technique = TechniqueSelectorView()
        self.modifier = ModifierSelectorView()
        self.provider = ProviderSelectorView()
        self.layout = pn.Column(
            self.technique,
            pn.Spacer(height=5),
            self.modifier,
            pn.Spacer(height=5),
            self.provider,
            pn.Spacer(height=5),
        )

    def __panel__(self) -> Viewable:
        return self.layout

    def set_provider_valid_api_callback(self, callback: Callable):
        self.provider.set_valid_api_key_callback(callback)

    def disable(self):
        self.provider.disable()
        self.technique.disable()
        self.modifier.disable()

    def enable(self):
        self.provider.enable()
        self.technique.enable()
        self.modifier.enable()
