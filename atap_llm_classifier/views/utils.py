from typing import Any, Callable

import panel as pn


def rx(obj: Any) -> tuple[pn.rx, Callable]:
    robj = pn.rx(obj)

    def set_obj(new: Any):
        robj.rx.value = new

    return robj, set_obj


def create_anchor_tag(link: str, content: str) -> str:
    return f"<a href={link} target='_blank'>{content}</a>"
