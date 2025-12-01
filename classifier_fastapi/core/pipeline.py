"""pipeline.py

Adapted pipeline for FastAPI - removes Corpus dependency and works with list[str].
Adds progress callbacks for job tracking and cost calculation.
"""

import asyncio
import inspect
from typing import Coroutine, Callable, Self

from litellm import RateLimitError, ModelResponse
from loguru import logger
from pydantic import BaseModel

from classifier_fastapi.core import core, config
from classifier_fastapi.core.models import LLMConfig
from classifier_fastapi.modifiers import Modifier, BaseModifier
from classifier_fastapi.providers.providers import LLMModelProperties
from classifier_fastapi.ratelimiters import TokenBucket
from classifier_fastapi import ratelimiters
from classifier_fastapi.techniques import Technique, BaseTechnique


class BatchResult(BaseModel):
    text_idx: int
    text: str
    classification: str
    prompt: str
    tokens_used: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    confidence: float | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None


class BatchResults(BaseModel):
    model: str
    technique: Technique
    user_schema: BaseModel
    modifier: Modifier
    llm_config: LLMConfig
    successes: list[BatchResult]
    fails: list[tuple[int, str]]
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None


def batch(
    texts: list[str],
    model_props: LLMModelProperties,
    llm_config: LLMConfig,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
    on_progress_callback: Callable | Coroutine | None = None,
) -> BatchResults:
    if config.mock.enabled:
        logger.info("Mock mode is enabled.")
    user_schema = technique.prompt_maker_cls.schema.model_validate(user_schema)
    res = asyncio.run(
        a_batch(
            texts,
            model_props,
            llm_config,
            technique,
            user_schema,
            modifier,
            on_progress_callback,
        ),
    )
    return res


class TaskContext(object):
    def __init__(
        self,
        text_idx: int | None,
        text: str | None,
    ):
        self.text_idx = text_idx
        self.text = text

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} text_idx={self.text_idx}>"

    def is_sentinel(self):
        return self.text_idx is None and self.text is None

    @classmethod
    def sentinel(cls) -> Self:
        return cls(None, None)


async def a_batch(
    texts: list[str],
    model_props: LLMModelProperties,
    llm_config: LLMConfig,
    technique: Technique,
    user_schema: BaseModel,
    modifier: Modifier,
    on_progress_callback: Callable | Coroutine | None = None,
) -> BatchResults:
    successes: list[BatchResult] = list()
    fails: list[tuple[int, str]] = list()
    total_tokens: int = 0

    cb_info: tuple[Callable | Coroutine, bool] | None = None
    if on_progress_callback is not None:
        cb_info = (on_progress_callback, inspect.iscoroutinefunction(on_progress_callback))

    prompt_maker: BaseTechnique = technique.get_prompt_maker(user_schema)
    mod_behaviour: BaseModifier = modifier.get_behaviour()

    rlimit_alg = config.RateLimiterAlg.TOKEN_BUCKET
    rlimits: ratelimiters.ProviderRateLimits = ratelimiters.get_rate_limits(model_props)
    logger.info(f"Rate Limit (request): {rlimits.requests}")
    logger.info(f"Rate Limit (tokens) : {rlimits.tokens}")

    rlimiter_reqs: TokenBucket = rlimit_alg.make_rate_limiter(rlimits.requests)
    rlimiter_toks: TokenBucket | None = None
    if rlimits.tokens is not None:
        rlimiter_toks = rlimit_alg.make_rate_limiter(rlimits.tokens)

    num_workers: int = config.batch.num_workers
    queue: asyncio.Queue[TaskContext] = asyncio.Queue(num_workers)

    async def worker(
        producer: asyncio.Queue[TaskContext],
        successes: list[BatchResult],
        fails: list[tuple[int, str]],
    ):
        nonlocal total_tokens
        while True:
            ctx: TaskContext = await producer.get()
            if ctx.is_sentinel():
                logger.info("Received sentinel value. Ending worker.")
                break

            logger.info(f"Consume task: {ctx}")
            retries_remaining: int = config.batch.rate_limit_max_retries
            exp_backoff_wait_s: float = (
                config.batch.rate_limit_retry_exp_backoff_first_wait_s
            )
            while retries_remaining >= 0:
                try:
                    res: core.ClassificationResult = await core.a_classify(
                        text=ctx.text,
                        model=model_props.name,
                        llm_config=llm_config,
                        technique=prompt_maker,
                        modifier=mod_behaviour,
                        api_key=model_props.api_key,
                        endpoint=model_props.endpoint,
                    )

                    # Extract token usage from response
                    tokens_used = None
                    prompt_tokens_val = None
                    completion_tokens_val = None
                    reasoning_tokens_val = None

                    if hasattr(res.response, 'usage') and res.response.usage:
                        usage = res.response.usage
                        tokens_used = usage.total_tokens

                        # Extract detailed token breakdown
                        if hasattr(usage, 'prompt_tokens'):
                            prompt_tokens_val = usage.prompt_tokens
                        if hasattr(usage, 'completion_tokens'):
                            completion_tokens_val = usage.completion_tokens

                        # Extract reasoning tokens if available (gpt-4.1 models)
                        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                            if hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                                reasoning_tokens_val = usage.completion_tokens_details.reasoning_tokens

                        # Only add if tokens_used is a valid number (not NaN or None)
                        import math
                        if tokens_used is not None and not (isinstance(tokens_used, float) and math.isnan(tokens_used)):
                            total_tokens += tokens_used

                    batch_res = BatchResult(
                        text_idx=ctx.text_idx,
                        text=ctx.text,
                        classification=res.classification,
                        prompt=res.prompt,
                        tokens_used=tokens_used,
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        reasoning_tokens=reasoning_tokens_val,
                        confidence=res.confidence,
                        reasoning=res.reasoning,
                        reasoning_content=res.reasoning_content,
                    )
                    successes.append(batch_res)

                    # Progress callback
                    if cb_info is not None:
                        cb, iscoro = cb_info
                        if iscoro:
                            await cb(len(successes), len(fails))
                        else:
                            cb(len(successes), len(fails))
                    break
                except RateLimitError as e:
                    logger.error(
                        f"Rate limit error on text idx: {ctx.text_idx}. Err - {e}"
                    )
                    logger.info(
                        f"Retry in {exp_backoff_wait_s} seconds... remaining={retries_remaining} text_idx={ctx.text_idx}."
                    )
                    await asyncio.sleep(exp_backoff_wait_s)
                    exp_backoff_wait_s *= 2
                    retries_remaining -= 1
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed classification on text idx: {ctx.text_idx}. Err - {e}"
                    )
                    fails.append((ctx.text_idx, str(e)))

                    # Progress callback for failures
                    if cb_info is not None:
                        cb, iscoro = cb_info
                        if iscoro:
                            await cb(len(successes), len(fails))
                        else:
                            cb(len(successes), len(fails))
                    break

    # Perform sanity checks on batch
    if not model_props.known_context_window():
        logger.warning(
            f"Skip batch context window check. Context window is not known for model: {model_props.name}"
        )
    elif not model_props.known_tokeniser():
        logger.warning(
            f"Skip batch context window check. Tokeniser is not known for model: {model_props.name}"
        )
    else:
        num_tokens: list[int] = list(
            map(model_props.count_tokens, map(prompt_maker.make_prompt, texts))
        )
        exceeded_indices = list(
            filter(
                lambda i: num_tokens[i] > model_props.context_window,
                range(len(num_tokens)),
            )
        )
        max_num_tokens = max(num_tokens)
        logger.info(
            f"Perform check: max number of tokens < context window for model: {model_props.name}."
        )
        if len(exceeded_indices) > 0:
            for i in exceeded_indices:
                logger.error(
                    f"Text {i} exceeds context window. {num_tokens[i]} > {model_props.context_window}"
                )
            raise RuntimeError(
                f"Max number of tokens in a text > context window for model {model_props.name}."
            )

        if rlimiter_toks is not None:
            logger.info("Perform check: max number of tokens < token rate limit.")

            if max_num_tokens > rlimiter_toks.capacity:
                raise RuntimeError("Max number of tokens > token rate limit.")

    worker_tasks = [
        asyncio.create_task(worker(queue, successes, fails)) for _ in range(num_workers)
    ]
    logger.info(f"Started {num_workers} workers to consume tasks.")

    rlimiter_reqs.start_replenisher()
    logger.info("Started requests rate limiter replenisher in the background.")
    if rlimiter_toks is not None:
        rlimiter_toks.start_replenisher()
        logger.info("Started tokens rate limiter replenisher in the background.")

    for i, text in enumerate(texts):
        release_coros = list()
        release_coros.append(await rlimiter_reqs.acquire(1))

        reqs_left, toks_left = None, None
        req_left: int = rlimiter_reqs.remaining
        if model_props.known_tokeniser() and rlimiter_toks is not None:
            prompt = prompt_maker.make_prompt(text)
            num_tokens = model_props.count_tokens(prompt)
            release_coros.append(await rlimiter_toks.acquire(num_tokens))
            toks_left = rlimiter_toks.remaining

        ctx = TaskContext(text_idx=i, text=text)
        await queue.put(ctx)
        logger.info(
            f"Produced task {i + 1}/{len(texts)}: {ctx}\t{req_left=} {toks_left=}"
        )

    logger.info("All tasks are produced. Waiting for all tasks to be consumed...")
    while not queue.empty():
        logger.info("Next check for all tasks consumed in 5s...")
        await asyncio.sleep(5)

    logger.info("All tasks are consumed. Cleaning up workers...")
    for _ in worker_tasks:
        await queue.put(TaskContext.sentinel())

    logger.info("Waiting for workers to complete...")
    await asyncio.gather(*worker_tasks)
    logger.info("All workers completed.")

    rlimiter_reqs.destroy()
    if rlimiter_toks is not None:
        rlimiter_toks.destroy()
    logger.info("Cleaned up rate limiter background replenishers.")

    # Aggregate detailed token counts
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0

    for success in successes:
        if success.prompt_tokens is not None:
            total_prompt_tokens += success.prompt_tokens
        if success.completion_tokens is not None:
            total_completion_tokens += success.completion_tokens
        if success.reasoning_tokens is not None:
            total_reasoning_tokens += success.reasoning_tokens

    # Calculate estimated cost if available
    estimated_cost = None
    import math
    if total_tokens > 0 and not (isinstance(total_tokens, float) and math.isnan(total_tokens)):
        from classifier_fastapi.core.cost import CostEstimator
        # Use actual token breakdown if available, otherwise estimate
        if total_prompt_tokens > 0 and total_completion_tokens > 0:
            estimated_cost = CostEstimator.calculate_actual_cost(
                input_tokens=total_prompt_tokens,
                output_tokens=total_completion_tokens,
                model=model_props.name,
            )
        else:
            # Fallback to estimation if detailed breakdown not available
            estimated_input_tokens = int(total_tokens * 0.8)
            estimated_output_tokens = int(total_tokens * 0.2)
            estimated_cost = CostEstimator.calculate_actual_cost(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                model=model_props.name,
            )

    return BatchResults(
        model=model_props.name,
        technique=technique,
        user_schema=user_schema,
        modifier=modifier,
        llm_config=llm_config,
        successes=successes,
        fails=fails,
        total_tokens=total_tokens,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        reasoning_tokens=total_reasoning_tokens,
        estimated_cost_usd=estimated_cost,
    )
