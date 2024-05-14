"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
from asyncio import Future

import litellm
from pydantic import BaseModel, Field
from loguru import logger

from atap_corpus import Corpus
from atap_corpus._types import Doc, Docs
from atap_llm_classifier.core import LLMModelConfig
from atap_llm_classifier.providers import LLMProvider
from atap_llm_classifier.techniques import Technique, NoTechnique, BaseTechnique
from atap_llm_classifier.modifiers import Modifier, NoModifier, BaseModifier
from atap_llm_classifier.utils import timeit
from atap_llm_classifier import core

litellm.set_verbose = False

MODEL = "ollama_chat/llama3"
API_BASE = "http://localhost:11434"
NUM_MESSAGES = 10


class PipelineResults(BaseModel):
    pass


def run(
    corpus: Corpus,
    provider: LLMProvider,
    technique: Technique | None = None,
    modifier: Modifier | None = None,
) -> PipelineResults:
    results = asyncio.run(a_run(corpus, provider, technique, modifier))
    print(results)
    return PipelineResults()


async def a_run(
    corpus: Corpus,
    technique: Technique | None = None,
    modifier: Modifier | None = None,
):
    docs: Docs = corpus[:1]
    print(docs)

    if technique is None:
        technique: BaseTechnique = NoTechnique()

    if modifier is None:
        modifier: BaseModifier = NoModifier()

    tasks = list()
    for doc in docs:
        task: Future = asyncio.create_task(
            core.a_classify(
                text=str(doc),
                model="gpt-3.5-turbo",
                llm_config=LLMModelConfig(),
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
        provider=LLMProvider.OPENAI,
    )
