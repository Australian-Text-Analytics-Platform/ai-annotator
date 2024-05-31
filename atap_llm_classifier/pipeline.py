"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
import inspect
from typing import Coroutine, Callable

from atap_corpus import Corpus
from atap_corpus._types import Docs
from loguru import logger
from pydantic import BaseModel

from atap_llm_classifier import core
from atap_llm_classifier.core import LLMConfig
from atap_llm_classifier.modifiers import Modifier, BaseModifier
from atap_llm_classifier.providers.providers import (
    LLMUserModelProperties,
    LLMProvider,
    LLMProviderUserProperties,
)
from atap_llm_classifier.ratelimiters import RateLimit
from atap_llm_classifier.techniques import Technique, BaseTechnique
from atap_llm_classifier import settings


class BatchResult(BaseModel):
    doc_idx: int
    classification_result: core.ClassificationResult


class BatchResults(BaseModel):
    corpus_name: str  # todo: Corpus is an arbitrary type, i suppose we can override serialise to use Corpus.serialise()
    model: str
    technique: Technique
    user_schema: BaseModel
    modifier: Modifier
    llm_config: LLMConfig
    results: list[BatchResult]


def batch(
    corpus: Corpus,
    provider: LLMProvider,
    model: str,
    api_key: str,
    llm_config: LLMConfig,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
    on_result_callback: Callable | Coroutine | None = None,
) -> BatchResults:
    logger.info("START run")
    user_provider_props: LLMProviderUserProperties = provider.get_user_properties(
        api_key=api_key
    )
    user_model: LLMUserModelProperties = user_provider_props.get_model_props(
        model=model
    )

    res = asyncio.run(
        a_batch(
            corpus,
            user_model,
            llm_config,
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
    user_model: LLMUserModelProperties,
    llm_config: LLMConfig,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
    on_result_callback: Callable | Coroutine | None = None,
) -> BatchResults:
    cb_info: tuple[Callable | Coroutine, bool] | None = None
    if on_result_callback is not None:
        cb_info = (on_result_callback, inspect.iscoroutinefunction(on_result_callback))

    docs: Docs = corpus.docs()

    prompt_maker: BaseTechnique = technique.get_prompt_maker(user_schema)
    mod_behaviour: BaseModifier = modifier.get_behaviour()

    coros: list[Coroutine] = [
        _a_classify_with_id(
            doc_idx=i,
            text=str(doc),
            model=user_model.name,
            api_key=user_model.validated_api_key.get_secret_value(),
            llm_config=llm_config,
            technique=prompt_maker,
            modifier=mod_behaviour,
        )
        for i, doc in enumerate(docs)
    ]

    batch_results: list[BatchResult] = list()
    coro: Coroutine

    rate_limit: RateLimit | None = settings.get_rate_limit(user_model)
    print(rate_limit)
    with settings.get_settings().RATE_LIMITER_ALG.get_rate_limiter()(
        on=coros,
        rate_limit=rate_limit,
        # rate_limit=RateLimit(max_requests=1, per_seconds=1),
    ) as (coros, semaphore):
        for coro in asyncio.as_completed(coros):
            doc_idx, classif_result = await coro
            batch_result: BatchResult = BatchResult(
                doc_idx=doc_idx,
                classification_result=classif_result,
            )
            batch_results.append(batch_result)
            if cb_info is not None:
                cb, iscoro = cb_info
                if iscoro:
                    await on_result_callback(batch_result)
                else:
                    on_result_callback(batch_result)

    return BatchResults(
        corpus_name=corpus.name,
        model=user_model.name,
        technique=technique,
        user_schema=user_schema,
        modifier=modifier,
        llm_config=llm_config,
        results=batch_results,
    )


async def _a_classify_with_id(
    doc_idx: int,
    **classify_kwargs,
) -> tuple[int, core.ClassificationResult]:
    res = await core.a_classify(**classify_kwargs)
    return doc_idx, res


if __name__ == "__main__":
    from atap_llm_classifier import config
    from atap_llm_classifier.techniques.schemas.zeroshot import (
        ZeroShotUserSchema,
        ZeroShotClass,
    )

    config.mock = True
    logger.info(f"Settings: {settings.get_settings()}")
    logger.info(f"Mock: {config.mock}")

    user_schema_ = ZeroShotUserSchema(
        classes=[ZeroShotClass(name="class 1", description="the first class")]
    )

    results_ = batch(
        corpus=Corpus([f"test sentence {i}" for i in range(1000)]),
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        api_key="",
        llm_config=LLMConfig(seed=42),
        user_schema=user_schema_,
        technique=Technique.ZERO_SHOT,
        modifier=Modifier.NO_MODIFIER,
        on_result_callback=lambda res: print("Got ", res.doc_idx),
    )
