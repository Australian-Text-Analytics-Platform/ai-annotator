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
from atap_llm_classifier.techniques import BaseTechnique
from atap_llm_classifier.models import (
    LLMModelConfig,
    LiteLLMMessage,
    LiteLLMArgs,
    LiteLLMRole,
)


class Result(BaseModel):
    text: str
    classification: str
    prompt: str
    response: ModelResponse


async def a_classify(
    text: str,
    model: str,
    llm_config: LLMModelConfig,
    technique: BaseTechnique,
    modifier: BaseModifier,
) -> Result:
    prompt = technique.make_prompt(text)
    prompt, llm_config = modifier.pre(prompt=prompt, llm_config=llm_config)

    # preconditions: technique, modifier applied to prompt and llm configs.
    msg = LiteLLMMessage(content=prompt, role=LiteLLMRole.USER)
    args = LiteLLMArgs(model=model, messages=[msg], stream=False)

    response: ModelResponse = await acompletion(**args.to_fn_args())

    classification: str = modifier.post(response=response)

    return Result(
        text=text,
        classification=classification,
        prompt=prompt,
        response=response,
    )
