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

        model_props = model_props.get_model_props(request.model)

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

        # Store cost
        if results.estimated_cost_usd or results.total_tokens:
            await job_manager.set_cost(job_id, {
                "total_usd": results.estimated_cost_usd,
                "total_tokens": results.total_tokens
            })

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

        model_props = provider.properties.get_model_props(request.model)
        user_schema = technique.prompt_maker_cls.schema.model_validate(request.user_schema)
        prompt_maker = technique.get_prompt_maker(user_schema)

        # Estimate tokens
        token_estimate = CostEstimator.estimate_tokens(
            request.texts,
            model_props,
            prompt_maker
        )

        warnings = []
        if token_estimate.get("warning"):
            warnings.append(token_estimate["warning"])

        # Estimate cost
        cost = None
        if token_estimate.get("total_tokens"):
            cost = CostEstimator.estimate_cost(
                token_estimate,
                request.provider,
                request.model
            )
            if cost is None:
                warnings.append(f"Pricing information not available for {request.provider}/{request.model}")

        return CostEstimateResponse(
            estimated_tokens=token_estimate.get("total_tokens", 0),
            estimated_cost_usd=cost,
            provider=request.provider,
            model=request.model,
            num_texts=len(request.texts),
            warnings=warnings
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider, model, or technique: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
