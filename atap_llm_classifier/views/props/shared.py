from pydantic import BaseModel


class ViewPropsWithName(BaseModel):
    name: str
    # todo: value mapper (dict of enum.name str -> str)


class ViewPropsWithNameDescription(ViewPropsWithName):
    description: str


class ViewPropsWithNameToolTip(ViewPropsWithName):
    tooltip: str
