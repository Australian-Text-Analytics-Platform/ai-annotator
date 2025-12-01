import io

import yaml
from loguru import logger

from atap_llm_classifier import errors
from atap_llm_classifier.formatter.models import OutputFormat
from atap_llm_classifier.settings import get_env_settings
from atap_llm_classifier.techniques.schemas import LLMoutputModel

__all__ = [
    "format_prompt",
    "unformat_output",
    "make_mock_response",
]


def format_prompt(
    prompt: str,
    output_keys: list[str],
) -> str:
    value_format = OutputFormat.infos().value_format
    output_format = get_env_settings().LLM_OUTPUT_FORMAT
    output_dict = {k: value_format.format(k) for k in output_keys}
    match output_format:
        case OutputFormat.YAML:
            output_format_instr: str = yaml.dump(output_dict)
        case _:
            raise NotImplementedError()
    templated: str = output_format.template.format_str.format(output_format_instr)
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
    output_format = get_env_settings().LLM_OUTPUT_FORMAT
    ptn = output_format.template.unformat_regex_compiled
    found = ptn.findall(string=llm_output)
    match len(found):
        case 0:
            raise errors.CorruptedLLMFormattedOutput(
                f"Corrupt LLM output format. Format={output_format}."
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
    output_dict = {}
    for k in output_keys:
        if k == "confidence":
            output_dict[k] = 0.95
        elif k in ["reasoning", "reason"]:
            output_dict[k] = "This is a mock reasoning explanation."
        else:
            output_dict[k] = f"This is a mock {k}."
    output_format = get_env_settings().LLM_OUTPUT_FORMAT
    match output_format:
        case OutputFormat.YAML:
            mock_res: str = yaml.safe_dump(output_dict)
        case _:
            raise NotImplementedError()
    return output_format.template.format_str.format(mock_res)
