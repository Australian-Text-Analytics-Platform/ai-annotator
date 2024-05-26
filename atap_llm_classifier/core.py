"""completions.py

This depends heavily on the litellm package.

Provide the core functions of classifications.

Note:
    performance of litellm's batch classification v.s. running multiple async is the equivalent.
"""

from litellm import acompletion, ModelResponse, Choices
from loguru import logger
from pydantic import BaseModel

import atap_llm_classifier as atap
from atap_llm_classifier.modifiers import BaseModifier
from atap_llm_classifier.techniques import BaseTechnique
from atap_llm_classifier import output_formatter, Settings, errors
from atap_llm_classifier.models import (
    LLMConfig,
    LiteLLMMessage,
    LiteLLMArgs,
    LiteLLMRole,
)

__all__ = ["a_classify", "Result"]


class Result(BaseModel):
    text: str
    classification: str
    prompt: str
    response: ModelResponse


async def a_classify(
    text: str,
    model: str,
    api_key: str,
    llm_config: LLMConfig,
    technique: BaseTechnique,
    modifier: BaseModifier,
    # todo: Add output format here
) -> Result:
    prompt: str = technique.make_prompt(text)
    prompt, llm_config = modifier.pre(prompt=prompt, llm_config=llm_config)
    prompt: str = output_formatter.format_prompt(
        prompt=prompt,
        output_keys=technique.template.output_keys,
    )

    # preconditions: technique, modifier applied to prompt and llm configs.
    msg = LiteLLMMessage(content=prompt, role=LiteLLMRole.USER)
    response: ModelResponse = await acompletion(
        **LiteLLMArgs(
            model=model,
            messages=[msg],
            stream=False,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            n=llm_config.n_completions,
            api_key=api_key,
        ).to_kwargs(),
        mock_response=output_formatter.make_mock_response(
            technique.template.output_model
        ),
    )

    unformatted_outputs: list[BaseModel | None] = list()
    choice: Choices
    for choice in response.choices:
        llm_output: str = choice.message.content
        try:
            unformatted = output_formatter.unformat_output(
                llm_output=llm_output,
                output_keys=technique.template.output_keys,
            )
            unformatted_outputs.append(unformatted)
        except errors.CorruptedLLMFormattedOutput as e:
            unformatted_outputs.append(None)
            logger.error(f"Corrupted LLM output format. Error: - {e}")

    classification: str = modifier.post(response=response)

    return Result(
        text=text,
        classification=classification,
        prompt=prompt,
        response=response,
    )
