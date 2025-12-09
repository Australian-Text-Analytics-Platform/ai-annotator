"""completions.py

This depends heavily on the litellm package.

Provide the core functions of classifications.

Note:
    performance of litellm's batch classification v.s. running multiple async is the equivalent.
"""

from litellm import acompletion, ModelResponse, Choices
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from loguru import logger
from pydantic import BaseModel

from atap_llm_classifier import errors, config
from atap_llm_classifier.formatter import formatter
from atap_llm_classifier.models import (
    LLMConfig,
    LiteLLMMessage,
    LiteLLMCompletionArgs,
    LiteLLMRole,
)
from atap_llm_classifier.modifiers import BaseModifier
from atap_llm_classifier.providers import LLMModelProperties
from atap_llm_classifier.techniques import BaseTechnique
from atap_llm_classifier.techniques.schemas import LLMoutputModel

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

    # Try with all parameters first, retry without unsupported ones if needed
    reasoning_effort_used = llm_config.reasoning_effort  # Track what was actually sent
    retry_count = 0
    max_retries = 2

    while retry_count <= max_retries:
        try:
            # Build completion args
            completion_args = LiteLLMCompletionArgs(
                model=model if not config.mock.enabled else "openai/gpt-3.5-turbo",
                messages=[msg],
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                n=llm_config.n_completions,
                stream=False,
                api_key=api_key,
                base_url=endpoint,
                reasoning_effort=reasoning_effort_used,
            )

            # On retry attempts, remove unsupported params
            kwargs = completion_args.to_kwargs()
            if retry_count == 1:
                # First retry: remove reasoning_effort
                kwargs.pop('reasoning_effort', None)
                reasoning_effort_used = None
            elif retry_count == 2:
                # Second retry: remove top_p and temperature too
                kwargs.pop('top_p', None)
                kwargs.pop('temperature', None)

            response: ModelResponse = await acompletion(
                **kwargs,
                mock_response=formatter.make_mock_response(output_keys)
                if config.mock.enabled
                else None,
            )
            break  # Success, exit loop

        except (BadRequestError, UnsupportedParamsError) as e:
            error_message = str(e)

            # Check if it's an unsupported parameter error
            if "reasoning_effort" in error_message.lower() and retry_count == 0:
                logger.warning(
                    f"Model {model} does not support reasoning_effort parameter. Retrying without it."
                )
                retry_count = 1
                continue

            elif ("top_p" in error_message.lower() or "temperature" in error_message.lower()) and retry_count <= 1:
                logger.warning(
                    f"Model {model} does not support top_p/temperature parameters. Retrying without them."
                )
                retry_count = 2
                continue
            else:
                # Re-raise if it's a different error or we've exhausted retries
                raise

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

    # Extract native reasoning_content from response (only if reasoning_effort was requested)
    reasoning_content = None
    if reasoning_effort_used and hasattr(response.choices[0].message, 'reasoning_content'):
        reasoning_content = response.choices[0].message.reasoning_content
        if reasoning_content:
            logger.info(f"Extracted reasoning_content from response (length: {len(reasoning_content)})")

    # Debug log if reasoning_effort was actually sent but no reasoning_content found
    if reasoning_effort_used and not reasoning_content:
        logger.info(
            f"reasoning_effort={reasoning_effort_used} was sent to model {model}, "
            f"but no reasoning_content found in response. This may be expected for models "
            f"that don't populate this field."
        )

    return ClassificationResult(
        text=text,
        classification=classification,
        prompt=prompt,
        response=response,
        confidence=confidence,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
    )
