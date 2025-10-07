"""completions.py

This depends heavily on the litellm package.

Provide the core functions of classifications.

Note:
    performance of litellm's batch classification v.s. running multiple async is the equivalent.
"""

from litellm import acompletion, ModelResponse, Choices
from loguru import logger
from pydantic import BaseModel

from classifier_fastapi import errors
from classifier_fastapi.core import config
from classifier_fastapi.formatter import formatter
from classifier_fastapi.core.models import (
    LLMConfig,
    LiteLLMMessage,
    LiteLLMCompletionArgs,
    LiteLLMRole,
)
from classifier_fastapi.modifiers import BaseModifier
from classifier_fastapi.providers import LLMModelProperties
from classifier_fastapi.techniques import BaseTechnique
from classifier_fastapi.techniques.schemas import LLMoutputModel

__all__ = [
    "a_classify",
    "ClassificationResult",
]


class ClassificationResult(BaseModel):
    text: str
    classification: str
    prompt: str
    response: ModelResponse


async def a_classify(
    text: str,
    model: str,
    llm_config: LLMConfig,
    technique: BaseTechnique,
    modifier: BaseModifier,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> ClassificationResult:
    prompt: str = technique.make_prompt(text)
    prompt, llm_config = modifier.pre(
        text=text,
        model=model,
        prompt=prompt,
        technique=technique,
        llm_config=llm_config,
    )
    prompt: str = formatter.format_prompt(
        prompt=prompt,
        output_keys=technique.template.output_keys,
    )

    # preconditions: technique, modifier, formatter applied to prompt and llm configs.
    msg = LiteLLMMessage(content=prompt, role=LiteLLMRole.USER)
    response: ModelResponse = await acompletion(
        **LiteLLMCompletionArgs(
            model=model if not config.mock.enabled else "openai/gpt-3.5-turbo",
            messages=[msg],
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            n=llm_config.n_completions,
            stream=False,
            api_key=api_key,
            base_url=endpoint,
        ).to_kwargs(),
        mock_response=formatter.make_mock_response(technique.template.output_keys)
        if config.mock.enabled
        else None,
    )

    unformatted_outputs: list[LLMoutputModel | None] = list()
    choice: Choices
    for i, choice in enumerate(response.choices):
        llm_output: str = choice.message.content
        try:
            unformatted = formatter.unformat_output(
                llm_output=llm_output,
                output_keys=technique.template.output_keys,
            )
            unformatted_outputs.append(unformatted)
        except errors.CorruptedLLMFormattedOutput as e:
            unformatted_outputs.append(None)
            logger.error(f"Corrupted LLM output format. Error: - {e}")
            logger.warning(
                f"None is added as output for choice {i}/{len(response.choices)}."
            )

    classification: str = modifier.post(
        response=response,
        outputs=unformatted_outputs,
        technique=technique,
        llm_config=llm_config,
        text=text,
        model=model,
    )

    return ClassificationResult(
        text=text,
        classification=classification,
        prompt=prompt,
        response=response,
    )
