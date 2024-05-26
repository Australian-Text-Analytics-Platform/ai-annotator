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
from loguru import logger
from pydantic import BaseModel

from atap_llm_classifier import core
from atap_llm_classifier.core import LLMConfig
from atap_llm_classifier.modifiers import Modifier, BaseModifier
from atap_llm_classifier.techniques import Technique, BaseTechnique

litellm.set_verbose = False


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
    from atap_llm_classifier.settings import get_settings
    from atap_llm_classifier.techniques.zeroshot import (
        ZeroShotUserSchema,
        ZeroShotClass,
    )

    os.environ["USE_MOCK"] = "true"
    os.environ["LLM_OUTPUT_FORMAT"] = "yaml"

    logger.info(f"Settings: {get_settings()}")

    user_schema_ = ZeroShotUserSchema(
        classes=[ZeroShotClass(name="class 1", description="the first class")]
    )

    results = run(
        corpus=Corpus([f"test sentence {i}" for i in range(3)]),
        model="gpt-3.5-turbo",
        api_key="",
        user_schema=user_schema_,
        technique=Technique.ZERO_SHOT,
        modifier=Modifier.NO_MODIFIER,
    )

    for res in results:
        print(res)
