import panel as pn

from techniques.techniques import Techniques


class TechniquesSelectorView(object):
    def __init__(self):
        self.selector = pn.widgets.Select(
            options=[t.name for t in Techniques],
        )
        # todo: once selected, pop down description, explanation and paper url
