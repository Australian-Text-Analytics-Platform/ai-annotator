"""jobs.py

Job status and management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException

from classifier_fastapi.api.models import (
    JobStatusResponse,
    JobProgress,
    ClassificationResultItem,
    JobStatus,
)
from classifier_fastapi.api.dependencies import get_current_api_key
from classifier_fastapi.job_manager import get_job_manager

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(get_current_api_key)
):
    """Get status and results of a classification job"""
    job_manager = get_job_manager()
    job = await job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Format results
    results = None
    if job.status == JobStatus.COMPLETED and job.results:
        results = [
            ClassificationResultItem(**result)
            for result in job.results
        ]

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        results=results,
        errors=job.errors if job.errors else None,
        cost=job.cost,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    api_key: str = Depends(get_current_api_key)
):
    """Cancel a running job"""
    job_manager = get_job_manager()
    job = await job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status {job.status}"
        )

    await job_manager.update_status(job_id, JobStatus.CANCELLED)

    return {"message": "Job cancelled successfully", "job_id": job_id}
