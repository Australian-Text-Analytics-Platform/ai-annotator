from pydantic import BaseModel


class SelectorProps(BaseModel):
    name: str
    # todo: value mapper (dict of enum.name str -> str)


class SelectorPropsWithDesc(SelectorProps):
    description: str


class SelectorPropsWithTooltip(SelectorProps):
    tooltip: str
