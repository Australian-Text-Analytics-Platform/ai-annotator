"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
from asyncio import Future
from typing import Sequence

import litellm
from loguru import logger
from atap_corpus import Corpus
from atap_corpus._types import Docs
from pydantic import BaseModel, SecretStr

from atap_llm_classifier import core
from atap_llm_classifier.core import LLMConfig
from atap_llm_classifier.modifiers import Modifier, BaseModifier
from atap_llm_classifier.techniques import Technique, BaseTechnique

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
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
) -> Sequence[core.Result]:
    logger.info("START run")
    res = asyncio.run(
        a_run(
            corpus,
            model,
            api_key,
            technique,
            user_schema,
            modifier,
        ),
    )
    logger.info("FINISH run")
    return res


async def a_run(
    corpus: Corpus,
    model: str,
    api_key: str,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
) -> Sequence[core.Result]:
    docs: Docs = corpus.docs()

    technique: BaseTechnique = technique.get_prompt_maker(user_schema)
    modifier: BaseModifier = modifier.get_behaviour()

    tasks = list()
    for doc in docs:
        logger.info(f"CREATE task: classify {str(doc)}")
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

    logger.info("WAIT for all tasks to finish.")
    results = await asyncio.gather(*tasks)
    # todo: callback -> classified.
    logger.info("FIN all tasks are finished.")
    return results


if __name__ == "__main__":
    import os

    os.environ["USE_MOCK"] = "true"
    os.environ["LLM_OUTPUT_FORMAT"] = "yaml"

    import atap_llm_classifier as atap

    logger.info(f"Settings: {atap.get_settings()}")

    from atap_llm_classifier.techniques.zeroshot import ZeroShotSchema, ZeroShotClass

    user_schema_ = ZeroShotSchema(
        classes=[ZeroShotClass(name="class 1", description="the first class")]
    )
    res = run(
        corpus=Corpus([f"test sentence {i}" for i in range(3)]),
        model="gpt-3.5-turbo",
        api_key="",
        user_schema=user_schema_,
        technique=Technique.ZERO_SHOT,
        modifier=Modifier.NO_MODIFIER,
    )
    print(res)

# # todo: do not use this - not implemented.
# async def a_run_multi_llm(
#     corpus: Corpus,
#     models: Sequence[str],
#     api_keys: Sequence[SecretStr],
#     technique: Technique | None = None,
#     modifier: Modifier | None = None,
# ):
#     # todo: make model, api_key, sys prompt, a BaseModel
#     #   allow multiple sys prompts to be used.
#     #
#     # todo: this basically runs a_run len(models) times.
#     #   then, just add an extra column that takes highest classified.
#     tasks = list()
#     for model, api_key in zip(models, api_keys):
#         task: Future = asyncio.create_task(
#             core.a_classify(
#                 text=corpus[:1],  # todo: use the corpus docs
#                 model=model,
#                 api_key=api_key,
#                 llm_config=LLMConfig(seed=42),
#                 technique=technique,
#                 modifier=modifier,
#             )
#         )
#         tasks.append(task)
#     results = await asyncio.gather(*tasks)
#     return results
