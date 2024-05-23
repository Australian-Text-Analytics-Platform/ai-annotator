from pydantic import BaseModel


class SelectorProps(BaseModel):
    name: str


class SelectorPropsWithDesc(SelectorProps):
    description: str


class SelectorPropsWithTooltip(SelectorProps):
    tooltip: str
