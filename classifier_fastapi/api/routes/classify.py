"""classify.py

Classification endpoints for batch classification and cost estimation.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from classifier_fastapi.api.models import (
    ClassificationRequest,
    CostEstimateRequest,
    JobCreateResponse,
    CostEstimateResponse,
    JobStatus,
)
from classifier_fastapi.api.dependencies import get_current_api_key
from classifier_fastapi.job_manager import get_job_manager
from classifier_fastapi.settings import get_settings

router = APIRouter(prefix="/classify", tags=["Classification"])


async def process_classification_job(job_id: str, request: ClassificationRequest):
    """Background task to process classification"""
    job_manager = get_job_manager()

    try:
        await job_manager.update_status(job_id, JobStatus.RUNNING)

        # Setup provider and model
        from classifier_fastapi.providers import LLMProvider
        from classifier_fastapi.techniques import Technique
        from classifier_fastapi.modifiers import Modifier
        from classifier_fastapi.core.models import LLMConfig
        from classifier_fastapi.core import pipeline

        provider = LLMProvider[request.provider.upper()]
        technique = Technique[request.technique.upper()]
        modifier = Modifier[request.modifier.upper()]

        model_props = provider.properties
        if request.llm_endpoint:
            model_props = model_props.with_endpoint(request.llm_endpoint)
        if request.llm_api_key:
            model_props = model_props.with_api_key(request.llm_api_key)

        try:
            model_props = model_props.get_model_props(request.model)
        except ValueError as e:
            # Better error message when model not found
            available_models = [m.id for m in model_props.models[:10]]  # Show first 10
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found or not accessible with provided API key. Available models: {', '.join(available_models)}... (showing first 10)"
            )

        llm_config = LLMConfig(
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0
        )

        # Validate user schema
        user_schema = technique.prompt_maker_cls.schema.model_validate(request.user_schema)

        # Progress callback
        async def on_progress(completed: int, failed: int):
            await job_manager.update_progress(job_id, completed, failed)

        # Run classification
        results = await pipeline.a_batch(
            texts=request.texts,
            model_props=model_props,
            llm_config=llm_config,
            technique=technique,
            user_schema=user_schema,
            modifier=modifier,
            on_progress_callback=on_progress
        )

        # Store results
        for success in results.successes:
            await job_manager.add_result(job_id, {
                "index": success.text_idx,
                "text": success.text,
                "classification": success.classification,
                "prompt": success.prompt,
            })

        for idx, error in results.fails:
            await job_manager.add_error(job_id, {
                "index": idx,
                "error": error
            })

        # Store cost with detailed breakdown
        if results.estimated_cost_usd or results.total_tokens:
            from classifier_fastapi.core.cost import CostEstimator

            cost_data = {
                "total_usd": results.estimated_cost_usd,
                "total_tokens": results.total_tokens,
            }

            # Add token breakdown only if available
            if results.prompt_tokens > 0:
                cost_data["input_tokens"] = results.prompt_tokens
            if results.completion_tokens > 0:
                cost_data["output_tokens"] = results.completion_tokens
            if results.reasoning_tokens > 0:
                cost_data["reasoning_tokens"] = results.reasoning_tokens

            # Calculate individual costs for each token type if breakdown available
            if results.prompt_tokens > 0 and results.completion_tokens > 0:
                pricing = CostEstimator.get_model_pricing(results.model)
                if pricing:
                    input_cost_per_token = pricing.get("input_cost_per_token", 0)
                    output_cost_per_token = pricing.get("completion_cost_per_token", 0)

                    cost_data["input_cost_usd"] = results.prompt_tokens * input_cost_per_token
                    cost_data["output_cost_usd"] = results.completion_tokens * output_cost_per_token

                    # Reasoning tokens typically billed at output rate or special rate
                    if results.reasoning_tokens > 0:
                        # Check if there's a specific reasoning token cost
                        reasoning_cost_per_token = pricing.get("reasoning_cost_per_token", output_cost_per_token)
                        cost_data["reasoning_cost_usd"] = results.reasoning_tokens * reasoning_cost_per_token

            await job_manager.set_cost(job_id, cost_data)

        await job_manager.update_status(job_id, JobStatus.COMPLETED)

    except Exception as e:
        await job_manager.add_error(job_id, {
            "error": str(e),
            "type": type(e).__name__
        })
        await job_manager.update_status(job_id, JobStatus.FAILED)


@router.post("/batch", response_model=JobCreateResponse)
async def create_classification_job(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_current_api_key)
):
    """
    Submit a batch classification job.

    Returns job_id for status tracking.
    """
    settings = get_settings()

    # Validate batch size
    if len(request.texts) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.texts)} exceeds maximum {settings.MAX_BATCH_SIZE}"
        )

    if len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")

    # Create job
    job_manager = get_job_manager()
    job_id = await job_manager.create_job(request.model_dump())

    # Start background task
    background_tasks.add_task(process_classification_job, job_id, request)

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Classification job created successfully",
        created_at=datetime.utcnow().isoformat()
    )


@router.post("/estimate-cost", response_model=CostEstimateResponse)
async def estimate_classification_cost(
    request: CostEstimateRequest,
    api_key: str = Depends(get_current_api_key)
):
    """
    Estimate the cost of a classification job before running it.

    Uses sample texts to calculate token counts.
    """
    from classifier_fastapi.core.cost import CostEstimator
    from classifier_fastapi.providers import LLMProvider
    from classifier_fastapi.techniques import Technique

    try:
        provider = LLMProvider[request.provider.upper()]
        technique = Technique[request.technique.upper()]

        user_schema = technique.prompt_maker_cls.schema.model_validate(request.user_schema)
        prompt_maker = technique.get_prompt_maker(user_schema)

        # Validate model exists (without API key, just check in general list)
        try:
            _ = provider.properties.get_model_props(request.model)
        except ValueError:
            available_models = [m.id for m in provider.properties.models[:10]]
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found for provider '{request.provider}'. Available models: {', '.join(available_models)}... (showing first 10)"
            )

        # Estimate tokens using LiteLLM token_counter
        token_estimate = CostEstimator.estimate_tokens(
            request.texts,
            request.model,
            prompt_maker
        )

        warnings = []
        if token_estimate.get("warning"):
            warnings.append(token_estimate["warning"])

        # Estimate cost using LiteLLM pricing
        cost = None
        pricing_per_million = None
        if token_estimate.get("total_tokens"):
            cost = CostEstimator.estimate_cost(
                token_estimate,
                request.model
            )
            if cost is None:
                warnings.append(f"Pricing information not available for {request.model}")
            else:
                pricing_per_million = CostEstimator.get_pricing_per_million(request.model)

        return CostEstimateResponse(
            estimated_tokens=token_estimate.get("total_tokens", 0),
            estimated_cost_usd=cost,
            provider=request.provider,
            model=request.model,
            num_texts=len(request.texts),
            input_tokens=token_estimate.get("input_tokens"),
            output_tokens=token_estimate.get("estimated_output_tokens"),
            input_cost_per_1m=pricing_per_million.get("input_per_million") if pricing_per_million else None,
            output_cost_per_1m=pricing_per_million.get("output_per_million") if pricing_per_million else None,
            warnings=warnings
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider, model, or technique: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
