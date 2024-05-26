import enum
import io
import re

import pydantic
import yaml
from loguru import logger
from pydantic import BaseModel, create_model, field_validator

import atap_llm_classifier as atap
from atap_llm_classifier.settings import Settings
from atap_llm_classifier.techniques.schemas import LLMoutputModel
from atap_llm_classifier.assets import Asset
from atap_llm_classifier import errors


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

    @classmethod
    def infos(cls) -> OutputFormatInfo:
        return OutputFormatInfo(**Asset.OUTPUT_FORMATTER.load())

    @property
    def template(self) -> OutputFormatTemplate:
        return getattr(self.infos().templates, self.value)


def format_prompt(
    prompt: str,
    output_keys: list[str],
) -> str:
    value_format = OutputFormat.infos().value_format
    settings: Settings = atap.get_settings()
    output_dict = {k: value_format.format(k) for k in output_keys}
    match settings.LLM_OUTPUT_FORMAT:
        case OutputFormat.YAML:
            output_format_instr: str = yaml.dump(output_dict)
        case _:
            raise NotImplementedError()
    templated: str = settings.LLM_OUTPUT_FORMAT.template.format_str.format(
        output_format_instr
    )
    return (
        prompt
        + "\n\n"
        + OutputFormat.infos().structure.format(
            instruction=OutputFormat.infos().instruction,
            templated=templated,
        )
    )


def unformat_output(
    llm_output: str,
    output_keys: list[str],
) -> LLMoutputModel:
    output_format = atap.get_settings().LLM_OUTPUT_FORMAT
    ptn = output_format.prompt_template.unformat_regex_compiled
    found = ptn.findall(string=llm_output)
    match len(found):
        case 0:
            raise errors.CorruptedLLMFormattedOutput(
                f"Expected LLM output format not found. Format={output_format}."
            )
        case 1:
            content: str = found[0]
        case _:
            logger.warning(
                "More than 1 instructed llm format output found. Using last output."
            )
            content: str = found[0]
    match output_format:
        case OutputFormat.YAML:
            unformatted_dict = yaml.safe_load(io.StringIO(content))
        case _:
            raise NotImplementedError()

    missing: list[str] = list()
    for k in output_keys:
        if k not in unformatted_dict:
            missing.append(k)
    if len(missing) > 0:
        raise errors.CorruptedLLMFormattedOutput(
            f"Incomplete output keys. Missing {','.join(missing)}. Required: {','.join(output_keys)}."
        )
    return LLMoutputModel(**unformatted_dict)


def make_mock_response(output_keys: list[str]) -> str:
    output_dict = {k: "This is a mock value for {}".format(k) for k in output_keys}
    output_format = atap.get_settings().LLM_OUTPUT_FORMAT
    match output_format:
        case OutputFormat.YAML:
            str_io = io.StringIO()
            yaml.safe_dump(output_dict, str_io)
            mock_res: str = str_io.read()
        case _:
            raise NotImplementedError()
    return mock_res
