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
    import json

    value_format = OutputFormat.infos().value_format
    output_format = get_env_settings().LLM_OUTPUT_FORMAT
    output_dict = {k: value_format.format(k) for k in output_keys}
    match output_format:
        case OutputFormat.YAML:
            output_format_instr: str = yaml.dump(output_dict)
        case OutputFormat.JSON:
            output_format_instr: str = json.dumps(output_dict, indent=2)
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
            # Fallback: Try to parse the raw output directly
            logger.warning(f"No code fence markers found. Attempting to parse raw output.")
            content: str = llm_output.strip()
        case 1:
            content: str = found[0]
        case _:
            logger.warning(
                "More than 1 instructed llm format output found. Using last output."
            )
            content: str = found[-1]
    match output_format:
        case OutputFormat.YAML:
            try:
                unformatted_dict = yaml.safe_load(io.StringIO(content))
            except yaml.YAMLError as e:
                # Log the problematic YAML for debugging
                logger.error(f"YAML parsing error: {e}")
                logger.error(f"Problematic YAML content:\n{content}")
                raise errors.CorruptedLLMFormattedOutput(
                    f"Invalid YAML format from LLM. Error: {e}"
                )
        case OutputFormat.JSON:
            import json
            try:
                unformatted_dict = json.loads(content)
            except json.JSONDecodeError as e:
                # Log the problematic JSON for debugging
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Problematic JSON content:\n{content}")
                raise errors.CorruptedLLMFormattedOutput(
                    f"Invalid JSON format from LLM. Error: {e}"
                )
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

    # Type coercion: Convert confidence from string to float if needed
    if 'confidence' in unformatted_dict and isinstance(unformatted_dict['confidence'], str):
        try:
            # Try direct conversion first
            unformatted_dict['confidence'] = float(unformatted_dict['confidence'])
        except ValueError:
            # If that fails, try parsing common text patterns like "0. nine five"
            import re
            conf_str = unformatted_dict['confidence'].lower()
            # Remove spaces and convert word numbers
            word_to_num = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
            }
            for word, num in word_to_num.items():
                conf_str = conf_str.replace(word, num)
            # Remove extra spaces
            conf_str = re.sub(r'\s+', '', conf_str)
            try:
                unformatted_dict['confidence'] = float(conf_str)
            except ValueError:
                logger.warning(f"Could not parse confidence value: {unformatted_dict['confidence']}")
                unformatted_dict['confidence'] = None  # Set to None if parsing fails

    return LLMoutputModel(**unformatted_dict)


def make_mock_response(output_keys: list[str]) -> str:
    import json

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
        case OutputFormat.JSON:
            mock_res: str = json.dumps(output_dict, indent=2)
        case _:
            raise NotImplementedError()
    return output_format.template.format_str.format(mock_res)
