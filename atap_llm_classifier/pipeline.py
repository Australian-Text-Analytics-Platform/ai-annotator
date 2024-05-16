"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
from asyncio import Future
from typing import Sequence

import litellm
from atap_corpus import Corpus
from atap_corpus._types import Docs
from pydantic import BaseModel, SecretStr

from atap_llm_classifier import core
from atap_llm_classifier.core import LLMConfig
from atap_llm_classifier.modifiers import Modifier, NoModifier, BaseModifier
from atap_llm_classifier.techniques import Technique, NoTechnique, BaseTechnique

litellm.set_verbose = False

MODEL = "ollama_chat/llama3"
API_BASE = "http://localhost:11434"
NUM_MESSAGES = 10


class UserInput(BaseModel):
    model: str
    api_key: SecretStr


class PipelineResults(BaseModel):
    pass


def run(
    corpus: Corpus,
    model: str,
    api_key: str,
    technique: Technique | None = None,
    modifier: Modifier | None = None,
) -> Sequence[core.Result]:
    loop = asyncio.get_event_loop()
    task = asyncio.run_coroutine_threadsafe(
        a_run(
            corpus,
            model,
            api_key,
            technique,
            modifier,
        ),
        loop,
    )
    return task.result()


async def a_run(
    corpus: Corpus,
    model: str,
    api_key: str,
    technique: Technique | None = None,
    modifier: Modifier | None = None,
) -> Sequence[core.Result]:
    docs: Docs = corpus[:1]

    if technique is None:
        technique: BaseTechnique = NoTechnique()

    if modifier is None:
        modifier: BaseModifier = NoModifier()

    tasks = list()
    for doc in docs:
        task: Future = asyncio.create_task(
            core.a_classify(
                text=str(doc),
                model=model,
                api_key=api_key,
                llm_config=LLMConfig(seed=42),
                technique=technique,
                modifier=modifier,
            )
        )
        # todo: callback -> classifying...
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    # todo: callback -> classified.
    return results


async def a_run_multi_llm(
    corpus: Corpus,
    models: Sequence[str],
    api_keys: Sequence[SecretStr],
    technique: Technique | None = None,
    modifier: Modifier | None = None,
):
    # todo: make model, api_key, sys prompt, a BaseModel
    #   allow multiple sys prompts to be used.
    #
    # todo: this basically runs a_run len(models) times.
    #   then, just add an extra column that takes highest classified.
    tasks = list()
    for model, api_key in zip(models, api_keys):
        task: Future = asyncio.create_task(
            core.a_classify(
                text=corpus[:1],  # todo: use the corpus docs
                model=model,
                api_key=api_key,
                llm_config=LLMConfig(seed=42),
                technique=technique,
                modifier=modifier,
            )
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    run(
        corpus=Corpus(["text"]),
        model="gpt-3.5-turbo",
        api_key="",
    )
