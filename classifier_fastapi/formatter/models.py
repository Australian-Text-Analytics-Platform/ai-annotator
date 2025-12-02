import enum
import re

import pydantic
from pydantic import BaseModel, field_validator

from classifier_fastapi.assets import Asset


class OutputFormatTemplate(BaseModel):
    format_str: str
    unformat_regex: str

    @property
    def unformat_regex_compiled(self) -> re.Pattern:
        return re.compile(self.unformat_regex, flags=re.DOTALL)

    @field_validator("unformat_regex", mode="before")
    @classmethod
    def is_regex(cls, v: str):
        try:
            re.compile(v)
            return v
        except Exception as e:
            raise pydantic.ValidationError("unformat_regex must be a valid regex.")

    @field_validator("unformat_regex", mode="after")
    @classmethod
    def must_have_one_group(cls, v: str):
        regex = re.compile(v)
        match regex.groups:
            case 1:
                return v
            case _:
                raise pydantic.ValidationError(
                    "unformat_regex must have 1 regex group."
                )


class OutputFormatTemplates(BaseModel):
    yaml: OutputFormatTemplate
    json: OutputFormatTemplate


class OutputFormatInfo(BaseModel):
    instruction: str
    value_format: str
    templates: OutputFormatTemplates
    structure: str

    @field_validator("value_format", mode="before")
    @classmethod
    def must_have_one_bracket(cls, v: str):
        found = re.findall(pattern=r"{}", string=v, flags=re.DOTALL)
        match len(found):
            case 1:
                return v
            case _:
                raise pydantic.ValidationError(
                    "value_format must have exactly 1 {} for arg formatting."
                )


class OutputFormat(enum.Enum):
    YAML: str = "yaml"
    JSON: str = "json"

    @classmethod
    def infos(cls) -> OutputFormatInfo:
        return OutputFormatInfo(**Asset.FORMATTER.load())

    @property
    def template(self) -> OutputFormatTemplate:
        return getattr(self.infos().templates, self.value)
