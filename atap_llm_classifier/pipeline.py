"""pipeline.py

Defines the steps of the classification pipeline.
"""

import asyncio
from enum import Enum

from pydantic import BaseModel, Field
from loguru import logger

from litellm import batch_completion, completion, acompletion
from litellm import batch_completion_models_all_responses

from atap_corpus import Corpus
from atap_corpus._types import Doc
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.providers import LLMProvider

__all__ = [
    "run_batch",
]

from atap_llm_classifier.utils import timeit


class LiteLLMRole(Enum):
    USER: str = "user"
    SYS: str = "system"
    ASSISTANT: str = "assistant"


class LiteLLMMessage(BaseModel):
    content: str = Field(frozen=True)
    role: LiteLLMRole | None = None


def make_message(
    doc: Doc,
    sys_msg: str | None,
) -> LiteLLMMessage:
    return LiteLLMMessage(user=str(doc), sys=sys_msg)


class Results(BaseModel):
    pass


@timeit
def run_batch(
    corpus: Corpus,
    technique: Technique,
    modifier: Modifier | None = None,
) -> Results:
    # expects corpus is loaded.
    messages = [[{"role": "user", "content": "good morning? "}] * 10_000]
    res = batch_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        mock_response="a mock response.",
        stream=False,
    )
    logger.info(res)
    return Results()


async def a_run_batch(
    corpus: Corpus,
    technique: Technique,
    modifier: Modifier | None = None,
) -> Results:
    # expects corpus is loaded.
    messages = [[{"role": "user", "content": "good morning? "}] * 10_000]

    tasks = [
        asyncio.create_task(
            acompletion(
                model="gpt-3.5-turbo", messages=msg, mock_response="mock response"
            )
        )
        for msg in messages
    ]

    await asyncio.gather(*tasks)
    return Results()


@timeit
def start_a_run_batch(*args, **kwargs):
    _ = asyncio.run(a_run_batch(*args, **kwargs))


if __name__ == "__main__":
    args = (
        Corpus(["text"]),
        Technique.CHAIN_OF_THOUGHT,
        Modifier.SELF_CONSISTENCY,
    )
    run_batch(*args)
    start_a_run_batch(*args)
