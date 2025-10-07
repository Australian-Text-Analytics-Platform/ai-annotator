"""
Health Check Routes

Simple health and readiness endpoints.
"""
from fastapi import APIRouter
from classifier_fastapi.api.models import HealthResponse
import time

router = APIRouter(prefix="/health", tags=["Health"])

_start_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="ATAP LLM Classifier API",
        version="0.1.0",
        uptime_seconds=time.time() - _start_time
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready"}
