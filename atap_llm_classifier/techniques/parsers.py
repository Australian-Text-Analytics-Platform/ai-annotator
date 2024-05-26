import enum
import io
from typing import TypeVar

import yaml

import atap_llm_classifier as atap
from atap_llm_classifier.techniques.templates import (
    BasePromptTemplate,
    PromptTemplateOutputFormats,
)
from atap_llm_classifier.assets import Asset


class OutputFormat(enum.StrEnum):
    YAML: str = "yaml"

    @property
    def template(self) -> str:
        return Asset.PARSER_TEMPLATES.get(self.value)


def make_output_format_for_prompt(
    out_formats_templates: PromptTemplateOutputFormats,
) -> str:
    match atap.get_settings().LLM_OUTPUT_FORMAT:
        case OutputFormat.YAML:
            return OutputFormat.YAML.template.format(out_formats_templates.yaml)


def parse_yaml(llm_out: str, out_format: PromptTemplateOutputFormats):
    pass


def make_mock_from_settings(out_formats_templates: PromptTemplateOutputFormats) -> str:
    match atap.get_settings().LLM_OUTPUT_FORMAT:
        case OutputFormat.YAML:
            struct: dict = yaml.safe_load(io.StringIO(out_formats_templates.yaml))
            for k in struct.keys():
                struct[k] = f"mock value for {k}"
            return OutputFormat.YAML.template.format(yaml.dump(struct))
