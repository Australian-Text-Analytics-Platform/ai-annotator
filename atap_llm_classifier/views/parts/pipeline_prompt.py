import panel as pn
from panel.viewable import Viewer, Viewable

from atap_llm_classifier.techniques import Technique


class PipelinePromptTab(Viewer):
    def __init__(self, **params):
        super().__init__(**params)

        self.live_edit = None
        self.preview = None

        self.layout = pn.Row(
            self.live_edit,
            self.preview,
        )

    def __panel__(self) -> Viewable:
        return self.layout


def create_live_edit(technique: Technique) -> Viewable:
    match technique:
        case Technique.ZERO_SHOT:
            raise NotImplementedError()
        case Technique.CHAIN_OF_THOUGHT:
            from atap_llm_classifier.techniques.schemas.cot import (
                CoTClass,
                CoTExample,
            )
            # todo: create text input based on the classes and examples


        case _:
            raise NotImplementedError()


def create_preview() -> Viewable:
    pass
