"""completions.py

This depends heavily on the litellm package.

Provide the core functions of classifications.

Note:
    performance of litellm's batch classification v.s. running multiple async is the equivalent.
"""

from litellm import acompletion, ModelResponse, Choices
from loguru import logger
from pydantic import BaseModel

from classifier_fastapi.core import errors
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
    confidence: float | None = None  # From LLM structured output
    reasoning: str | None = None  # From LLM structured output (prompted via enable_reasoning)
    reasoning_content: str | None = None  # From response.choices[0].message.reasoning_content (native LiteLLM reasoning mode)


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
    # Use dynamic output_keys from technique (includes reasoning if enabled)
    output_keys = technique.output_keys if hasattr(technique, 'output_keys') else technique.template.output_keys

    prompt: str = formatter.format_prompt(
        prompt=prompt,
        output_keys=output_keys,
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
            reasoning_effort=llm_config.reasoning_effort,
        ).to_kwargs(),
        mock_response=formatter.make_mock_response(output_keys)
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
                output_keys=output_keys,
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

    # Extract confidence and reasoning from unformatted_outputs
    confidence = None
    reasoning = None
    if unformatted_outputs and unformatted_outputs[0] is not None:
        output = unformatted_outputs[0]
        confidence = getattr(output, 'confidence', None)
        reasoning = getattr(output, 'reasoning', None)
        # For CoT, try 'reason' as well
        if reasoning is None:
            reasoning = getattr(output, 'reason', None)

        # Truncate reasoning at 110% of max_chars
        if reasoning and technique.enable_reasoning:
            max_len = int(technique.max_reasoning_chars * 1.1)
            if len(reasoning) > max_len:
                reasoning = reasoning[:max_len]

    # Extract native reasoning_content from response
    reasoning_content = None
    if hasattr(response.choices[0].message, 'reasoning_content'):
        reasoning_content = response.choices[0].message.reasoning_content

    return ClassificationResult(
        text=text,
        classification=classification,
        prompt=prompt,
        response=response,
        confidence=confidence,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
    )
