"""completions.py

This depends heavily on the litellm package.

Provide the functions to call the LLM.
"""

import asyncio

from litellm import acompletion, ModelResponse


def create_litellm_msg(msg: str) -> list[dict]:
    return [{"content": msg, "role": "user"}]


async def a_completion(model: str, msg: str) -> ModelResponse:
    messages = create_litellm_msg(msg)
    res = await acompletion(model=model, messages=messages)
    return res


async def a_batch_completions(model: str, msgs: list[str]) -> list[ModelResponse]:
    tasks = [asyncio.create_task(a_completion(model=model, msg=msg)) for msg in msgs]
    results: list[ModelResponse]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return list(results)


def batch_completions(
    model: str,
    msgs: list[str],
) -> list[ModelResponse]:
    results = asyncio.run(a_batch_completions(model, msgs))
    return results
