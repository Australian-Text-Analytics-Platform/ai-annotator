"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
from asyncio import Future

import litellm
from atap_corpus import Corpus
from atap_corpus._types import Docs
from pydantic import BaseModel, SecretStr

from atap_llm_classifier import core
from atap_llm_classifier.core import LLMModelConfig
from atap_llm_classifier.modifiers import Modifier, NoModifier, BaseModifier
from atap_llm_classifier.providers import LLMProvider
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
) -> PipelineResults:
    results = asyncio.run(
        a_run(
            corpus,
            model,
            api_key,
            technique,
            modifier,
        )
    )
    print(results)
    return PipelineResults()


async def a_run(
    corpus: Corpus,
    model: str,
    api_key: str,
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
                model=model,
                api_key=api_key,
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
        model="gpt-3.5-turbo",
        api_key="",
    )
