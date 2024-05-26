import panel as pn
from panel.viewable import Viewer, Viewable


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
