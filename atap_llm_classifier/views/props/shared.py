from pydantic import BaseModel


class ViewPropsWithName(BaseModel):
    name: str


class ViewPropsWithNameDescription(ViewPropsWithName):
    description: str


class ViewPropsWithNameToolTip(ViewPropsWithName):
    tooltip: str


class ViewPropsWithNameWidth(ViewPropsWithName):
    width: int
