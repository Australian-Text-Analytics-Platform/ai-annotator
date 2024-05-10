"""pipeline.py

Defines the steps of the classification pipeline.
"""

import asyncio
from asyncio import Future
from enum import Enum

import litellm
from pydantic import BaseModel, Field
from loguru import logger

from litellm import batch_completion, completion, acompletion

from atap_corpus import Corpus
from atap_corpus._types import Doc
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.modifiers import Modifier

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


litellm.set_verbose = False

MODEL = "ollama_chat/llama3"
API_BASE = "http://localhost:11434"
NUM_MESSAGES = 10


async def a_run_batch(
    corpus: Corpus,
    technique: Technique,
    modifier: Modifier | None = None,
) -> Results:
    # expects corpus is loaded.
    message = {"role": "user", "content": "good morning? "}
    messages = [[message]] * NUM_MESSAGES
    # technique.modify_prompt(corpus.docs())

    # modifier to modify ModelArgs

    tasks = [
        asyncio.create_task(
            acompletion(
                model=MODEL,
                api_base=API_BASE,
                messages=msg,
                mock_response="mock response",
            )
        )
        for msg in messages
    ]

    # modifiers need to modify the result classifications

    results: tuple[Future] = await asyncio.gather(*tasks)
    for res in results:
        logger.info(res)
    return Results()


@timeit
def start_a_run_batch(*args, **kwargs):
    _ = asyncio.run(a_run_batch(*args, **kwargs))


@timeit
def run_batch(
    corpus: Corpus,
    technique: Technique,
    modifier: Modifier | None = None,
) -> Results:
    # expects corpus is loaded.
    message = {"role": "user", "content": "good morning? "}
    messages = [[message]] * NUM_MESSAGES
    results = batch_completion(
        model=MODEL,
        api_base=API_BASE,
        messages=messages,
        # mock_response="a mock response.",
        stream=False,
    )

    for res in results:
        logger.info(res)
    return Results()


if __name__ == "__main__":
    args = (
        Corpus(["text"]),
        Technique.CHAIN_OF_THOUGHT,
        Modifier.SELF_CONSISTENCY,
    )
    start_a_run_batch(*args)
    run_batch(*args)
    start_a_run_batch(*args)
