import html
from collections import namedtuple

import panel as pn
from panel.viewable import Viewer, Viewable
from pydantic import BaseModel

from atap_llm_classifier.views.props import ViewProp, PipePromptProps
from atap_llm_classifier.views.settings import get_settings
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.techniques.schemas import (
    CoTClass,
    CoTExample,
    ZeroShotClass,
    ZeroShotUserSchema,
)

LIVE_UPDATE: bool = get_settings().PIPE_PROMPT_LIVE_UPDATE
text_input_key: str = "value_input" if LIVE_UPDATE else "value"

props: PipePromptProps = ViewProp.PIPE_PROMPT.properties


class PipelinePrompt(Viewer):
    def __init__(self, technique: Technique, **params):
        super().__init__(**params)
        self.technique: Technique = technique
        self.user_schema_rx = pn.rx(create_dummy_user_schema(self.technique))

        self.live_edit = create_live_edit(
            technique=self.technique,
            user_schema_rx=self.user_schema_rx,
        )
        self.prompt = pn.pane.Str(
            self.user_schema_rx.rx.pipe(self.technique.get_prompt_maker)
            .rx.pipe(
                lambda maker: maker.make_prompt(
                    text=props.prompt_preview.text_placeholder
                )
            )
            .rx.pipe(html.escape)
        )
        self.preview = pn.Column(
            pn.pane.Markdown(f"## {props.prompt_preview.name}"),
            self.prompt,
            width=500,
        )

        self.layout = pn.Row(
            self.live_edit,
            pn.Spacer(width=20),
            self.preview,
            sizing_mode="stretch_both",
        )

    def __panel__(self) -> Viewable:
        return self.layout

    @property
    def user_schema(self) -> BaseModel:
        return self.user_schema_rx.rx.value

    def disable(self):
        pass


def create_dummy_user_schema(technique: Technique) -> BaseModel:
    match technique:
        case Technique.ZERO_SHOT:
            from atap_llm_classifier.techniques.schemas import ZeroShotUserSchema

            return ZeroShotUserSchema(classes=[ZeroShotClass(name="", description="")])
        case Technique.CHAIN_OF_THOUGHT:
            # todo: return CoTUserSchema but also means we need live edit implementation too.
            raise NotImplementedError()
        case _:
            raise NotImplementedError()


def create_live_edit(technique: Technique, user_schema_rx) -> Viewable:
    match technique:
        case Technique.ZERO_SHOT:
            Row = namedtuple("Row", ["widget", "name", "description", "plus", "minus"])

            def new_row() -> Row:
                name_title = ZeroShotClass.schema()["properties"]["name"]["title"]
                desc_title = ZeroShotClass.schema()["properties"]["description"][
                    "title"
                ]

                name_inp = pn.widgets.TextInput(
                    name=name_title + ":", height=50, width=120
                )
                desc_inp = pn.widgets.TextInput(
                    name=desc_title + ":",
                    height=name_inp.height,
                    sizing_mode="stretch_width",
                )

                plus = pn.widgets.Button(
                    name="+",
                    width=25,
                    height=int(name_inp.height * 0.6),
                    margin=(10, 5),
                )
                minus = pn.widgets.Button(
                    name="-", width=plus.width, height=plus.height, margin=plus.margin
                )

                row = pn.layout.Row(
                    name_inp, desc_inp, plus, minus, height=name_inp.height, margin=5
                )
                return Row(
                    widget=row,
                    name=name_inp,
                    description=desc_inp,
                    plus=plus,
                    minus=minus,
                )

            rows = list()
            classes = pn.Column(
                sizing_mode="stretch_both",
            )

            live_edit = pn.Column(
                pn.pane.Markdown(f"## {props.live_edit.classes.name}"),
                classes,
                sizing_mode="stretch_both",
            )

            def update(*args):
                zshot_classes = list(
                    map(
                        lambda r: ZeroShotClass(
                            name=r.name.value_input,
                            description=r.description.value_input,
                        ),
                        rows,
                    )
                )
                user_schema_rx.rx.value = ZeroShotUserSchema(classes=zshot_classes)

            def insert_new_row(idx: int):
                row = new_row()
                rows.append(row)
                classes.insert(idx, row.widget)
                update(row.name)
                row.name.param.watch(update, text_input_key)
                row.description.param.watch(update, text_input_key)

                def pop_row(*args):
                    classes.remove(row.widget)
                    rows.remove(row)
                    update()

                row.minus.on_click(pop_row)
                row.plus.on_click(
                    lambda *args: insert_new_row(classes.index(row.widget) + 1)
                )
                return row

            row = insert_new_row(0)
            row.minus.disabled = True
            return live_edit
        case Technique.CHAIN_OF_THOUGHT:
            raise NotImplementedError()
        case _:
            raise NotImplementedError()
