import enum
from typing import TypeVar

import atap_llm_classifier as atap
from atap_llm_classifier.techniques.templates import (
    BasePromptTemplate,
    PromptTemplateOutputFormats,
)

Parsed = TypeVar("Parsed", bound=BasePromptTemplate)


class OutputFormat(enum.StrEnum):
    YAML: str = "yaml"

    @property
    def template(self) -> str:
        match self:
            case OutputFormat.YAML:
                return """
```yaml
{}
```
"""


def make_output_format(model: PromptTemplateOutputFormats) -> str:
    match atap.get_settings().LLM_OUTPUT_FORMAT:
        case OutputFormat.YAML:
            return OutputFormat.YAML.template.format(model.yaml)


def parse_yaml(llm_out: str, model: Parsed) -> Parsed:
    # todo:
    pass


def parse_json(llm_out: str, model: Parsed) -> Parsed:
    # todo:
    pass
