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
from atap_llm_classifier.ratelimiters import RateLimit, TokenBucket
from atap_llm_classifier.settings import (
    ProviderRateLimits,
    get_settings,
    get_rate_limits,
)
from atap_llm_classifier.techniques import Technique, BaseTechnique


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


class TaskContext(object):
    def __init__(
        self,
        doc_idx: int,
        text: str,
        release_coros: list[Coroutine],
    ):
        self.doc_idx = doc_idx
        self.text = text
        self.release_coros = release_coros

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(doc_idx={self.doc_idx}, #release_coros={len(self.release_coros)})"


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
    batch_results: list[BatchResult] = list()

    cb_info: tuple[Callable | Coroutine, bool] | None = None
    if on_result_callback is not None:
        cb_info = (on_result_callback, inspect.iscoroutinefunction(on_result_callback))

    docs: Docs = corpus.docs()

    prompt_maker: BaseTechnique = technique.get_prompt_maker(user_schema)
    mod_behaviour: BaseModifier = modifier.get_behaviour()

    rlimit_alg = get_settings().RATE_LIMITER_ALG
    rlimits: ProviderRateLimits = get_rate_limits(user_model)
    logger.info(f"Rate Limit (request): {rlimits.requests}")
    logger.info(f"Rate Limit (tokens) : {rlimits.tokens}")

    rlimiter_reqs: TokenBucket = rlimit_alg.make_rate_limiter(rlimits.requests)
    rlimiter_toks: TokenBucket | None = None
    if rlimits.tokens is not None:
        rlimiter_toks = rlimit_alg.make_rate_limiter(rlimits.tokens)

    num_workers: int = 3  # todo: settings
    queue: asyncio.Queue[TaskContext | None] = asyncio.Queue(num_workers)

    async def worker(
        queue: asyncio.Queue[TaskContext | None],
        batch_results: list[BatchResult],
    ):
        while True:
            ctx: TaskContext | None = await queue.get()
            if ctx is None:
                break
            res: core.ClassificationResult = await core.a_classify(
                text=str(doc),
                model=user_model.name,
                api_key=user_model.validated_api_key.get_secret_value(),
                llm_config=llm_config,
                technique=prompt_maker,
                modifier=mod_behaviour,
            )
            for release_fn in ctx.release_coros:
                await release_fn()

            batch_res = BatchResult(doc_idx=ctx.doc_idx, classification_result=res)
            batch_results.append(batch_res)
            if cb_info is not None:
                cb, iscoro = cb_info
                if iscoro:
                    await on_result_callback(batch_res)
                else:
                    on_result_callback(batch_res)

    worker_tasks = [
        asyncio.create_task(worker(queue, batch_results)) for _ in range(num_workers)
    ]
    logger.info(f"Started {num_workers} workers.")
    for i, doc in enumerate(docs):
        text = str(doc)
        release_coros = list()
        release_coros.append(await rlimiter_reqs.acquire(1))
        if rlimiter_toks is not None:
            prompt = prompt_maker.make_prompt(str(doc))
            num_tokens = user_model.count_tokens(prompt)
            release_coros.append(await rlimiter_toks.acquire(num_tokens))

        ctx = TaskContext(doc_idx=i, text=text, release_coros=release_coros)
        await queue.put(ctx)
        logger.info(f"Produced task {i + 1}/{len(docs)}: {ctx}")

    logger.info("All tasks are produced. Waiting until all tasks are consumed...")
    while not queue.empty():
        await asyncio.sleep(5)

    logger.info("All tasks are consumed. Cleaning up workers...")
    for _ in worker_tasks:
        await queue.put(None)

    logger.info("Waiting for workers to complete...")
    await asyncio.gather(*worker_tasks)
    logger.info("All workers completed.")

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
    logger.info(f"Settings: {get_settings()}")
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

"""
```python
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
```
"""
