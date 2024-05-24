"""completions.py

This depends heavily on the litellm package.

Provide the core functions of classifications.

Note:
    performance of litellm's batch classification v.s. running multiple async is the equivalent.
"""

import json
import enum
import asyncio
from asyncio import Future
from typing import Iterable

from litellm import acompletion, ModelResponse
from pydantic import BaseModel, Field

from atap_llm_classifier.modifiers import BaseModifier, NoModifier
from atap_llm_classifier.techniques import BaseTechnique, parsers
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
) -> Result:
    prompt: str = technique.make_prompt(text)
    prompt, llm_config = modifier.pre(prompt=prompt, llm_config=llm_config)

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
        mock_response=parsers.make_mock_from_settings(
            technique.template.output_formats
        ),
    )
    classification: str = modifier.post(response=response)

    return Result(
        text=text,
        classification=classification,
        prompt=prompt,
        response=response,
    )
