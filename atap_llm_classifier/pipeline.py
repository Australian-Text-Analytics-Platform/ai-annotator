"""pipeline.py

Defines the steps of the full pipeline.

The output
"""

import asyncio
import inspect
import random
from typing import Coroutine, Callable

import httpx
from atap_corpus import Corpus
from atap_corpus._types import Docs
from litellm import RateLimitError
from loguru import logger
from pydantic import BaseModel

from atap_llm_classifier import core, config
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
    successes: list[BatchResult]
    fails: list[tuple[int, str]]


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
    logger.info(f"START run. config.mock={config.mock}.")
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
    successes: list[BatchResult] = list()
    fails: list[tuple[int, str]] = list()

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

    num_workers: int = get_settings().BATCH_NUM_WORKERS
    queue: asyncio.Queue[TaskContext | None] = asyncio.Queue(num_workers)

    async def worker(
        queue: asyncio.Queue[TaskContext | None],
        successes: list[BatchResult],  # todo: potentially race condition
        fails: list[tuple[int, str]],  # todo: same here
    ):
        while True:
            ctx: TaskContext | None = await queue.get()
            if ctx is None:  # sentinel value
                logger.info("Received sentinel value. Ending worker.")
                break

            logger.info(f"Consume task: {ctx}")
            retries_remaining: int = get_settings().BATCH_RATE_LIMIT_MAX_RETRIES
            exp_backoff_wait_s: float = (
                get_settings().BATCH_RATE_LIMIT_RETRY_EXP_BACKOFF_FIRST_WAIT_S
            )
            while retries_remaining >= 0:
                try:
                    res: core.ClassificationResult = await core.a_classify(
                        text=ctx.text,
                        model=user_model.name,
                        api_key=user_model.validated_api_key.get_secret_value(),
                        llm_config=llm_config,
                        technique=prompt_maker,
                        modifier=mod_behaviour,
                    )

                    batch_res = BatchResult(
                        doc_idx=ctx.doc_idx, classification_result=res
                    )
                    successes.append(batch_res)
                    if cb_info is not None:
                        cb, iscoro = cb_info
                        if iscoro:
                            await on_result_callback(batch_res)
                        else:
                            on_result_callback(batch_res)
                    break
                except RateLimitError as e:
                    logger.error(
                        f"Rate limit error on doc idx: {ctx.doc_idx}. Err - {e}"
                    )
                    logger.info(
                        f"Retry in {exp_backoff_wait_s} seconds... remaining={retries_remaining} doc_idx={ctx.doc_idx}."
                    )
                    await asyncio.sleep(exp_backoff_wait_s)
                    exp_backoff_wait_s *= 2
                    retries_remaining -= 1
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed classification on doc idx: {ctx.doc_idx}. Err - {e}"
                    )
                    fails.append((ctx.doc_idx, str(e)))
                    break

            for release_fn in ctx.release_coros:
                await release_fn()
            # todo: put try-except-finally block on while True, finally return local successes, fails
            #   see notes in the args above.
            #   although async should be single threaded in python.

    # perform sanity checks
    max_num_tokens: int = max(
        map(user_model.count_tokens, map(prompt_maker.make_prompt, map(str, docs)))
    )
    logger.info(
        f"Sanity check: max number of tokens < context window for model {user_model.name}."
    )
    if max_num_tokens > user_model.context_window:
        raise RuntimeError(
            f"Max number of tokens > context window for model {user_model.name}."
        )

    if rlimiter_toks is not None:
        logger.info("Sanity check: max number of tokens < token rate limit.")

        if max_num_tokens > rlimiter_toks.capacity:
            raise RuntimeError("Max number of tokens > token rate limit.")

    worker_tasks = [
        asyncio.create_task(worker(queue, successes, fails)) for _ in range(num_workers)
    ]
    logger.info(f"Started {num_workers} workers to consume tasks.")
    for i, doc in enumerate(docs):
        text = str(doc)
        release_coros = list()
        release_coros.append(await rlimiter_reqs.acquire(1))

        reqs_left, toks_left = None, None
        req_left: int = rlimiter_reqs.remaining
        if rlimiter_toks is not None:
            prompt = prompt_maker.make_prompt(str(doc))
            num_tokens = user_model.count_tokens(prompt)
            release_coros.append(await rlimiter_toks.acquire(num_tokens))
            toks_left = rlimiter_toks.remaining

        ctx = TaskContext(doc_idx=i, text=text, release_coros=release_coros)
        await queue.put(ctx)
        logger.info(
            f"Produced task {i + 1}/{len(docs)}: {ctx}\t{req_left=} {toks_left=}"
        )

    logger.info("All tasks are produced. Waiting for all tasks to be consumed...")
    while not queue.empty():
        logger.info("Next check for all tasks consumed in 5s...")
        await asyncio.sleep(5)

    logger.info("All tasks are consumed. Cleaning up workers...")
    for _ in worker_tasks:
        await queue.put(None)

    logger.info("Waiting for workers to complete...")
    await asyncio.gather(*worker_tasks)
    logger.info("All workers completed.")

    rlimiter_reqs.destroy()
    rlimiter_toks.destroy()

    return BatchResults(
        corpus_name=corpus.name,
        model=user_model.name,
        technique=technique,
        user_schema=user_schema,
        modifier=modifier,
        llm_config=llm_config,
        successes=successes,
        fails=fails,
    )


if __name__ == "__main__":
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
