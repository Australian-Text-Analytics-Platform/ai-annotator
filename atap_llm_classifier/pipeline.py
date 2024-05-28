"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
import inspect
from typing import Coroutine, Callable

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


class BatchSingular(BaseModel):
    doc_idx: int
    classification_result: core.ClassificationResult


class BatchResults(BaseModel):
    # corpus: Corpus
    # model: str
    # technique: Technique
    # user_schema: BaseModel
    # modifier: Modifier
    # llm_config: LLMConfig
    results: list[BatchSingular]


def batch(
    corpus: Corpus,
    model: str,
    api_key: str,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
    on_result_callback: Callable | Coroutine | None = None,
) -> BatchResults:
    logger.info("START run")
    res = asyncio.run(
        a_batch(
            corpus,
            model,
            api_key,
            technique,
            user_schema,
            modifier,
            on_result_callback,
        ),
    )
    logger.info("FINISH run")
    return res


async def a_batch(
    corpus: Corpus,
    model: str,
    api_key: str,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
    on_result_callback: Callable | Coroutine | None = None,
) -> BatchResults:
    cb_info: tuple[Callable | Coroutine, bool] | None = None
    if on_result_callback is not None:
        cb_info = (on_result_callback, inspect.iscoroutine(on_result_callback))

    docs: Docs = corpus.docs()

    technique: BaseTechnique = technique.get_prompt_maker(user_schema)
    modifier: BaseModifier = modifier.get_behaviour()

    coros: list[Coroutine] = [
        _a_classify_with_id(
            doc_idx=i,
            text=str(doc),
            model=model,
            api_key=api_key,
            llm_config=LLMConfig(seed=42),
            technique=technique,
            modifier=modifier,
        )
        for i, doc in enumerate(docs)
    ]

    singulars: list[BatchSingular] = list()
    coro: Coroutine
    for coro in asyncio.as_completed(coros):
        doc_idx, classif_result = await coro
        singular: BatchSingular = BatchSingular(
            doc_idx=doc_idx,
            classification_result=classif_result,
        )
        singulars.append(singular)
        if cb_info is not None:
            cb, iscoro = cb_info
            if iscoro:
                await on_result_callback(singular)
            else:
                on_result_callback(singular)

    return BatchResults(
        # corpus=Corpus,
        results=singulars,
    )


async def _a_classify_with_id(
    doc_idx: int,
    **classify_kwargs,
) -> tuple[int, core.ClassificationResult]:
    res = await core.a_classify(**classify_kwargs)
    return doc_idx, res


if __name__ == "__main__":
    from atap_llm_classifier.settings import get_settings
    from atap_llm_classifier.techniques.zeroshot import (
        ZeroShotUserSchema,
        ZeroShotClass,
    )

    logger.info(f"Settings: {get_settings()}")

    user_schema_ = ZeroShotUserSchema(
        classes=[ZeroShotClass(name="class 1", description="the first class")]
    )

    from pprint import pprint

    results_ = batch(
        corpus=Corpus([f"test sentence {i}" for i in range(3)]),
        model="gpt-3.5-turbo",
        api_key="",
        user_schema=user_schema_,
        technique=Technique.ZERO_SHOT,
        modifier=Modifier.NO_MODIFIER,
        on_result_callback=lambda res: pprint(res.model_dump()),
    )

    for res in results_:
        print(res)
