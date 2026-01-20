"""main.py

FastAPI Main Application

Entry point for the classification API service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from loguru import logger

from classifier_fastapi.settings import get_settings
from classifier_fastapi.api.routes import health, classify, jobs, models
from classifier_fastapi.job_manager import get_job_manager

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.start_time = time.time()

    # Load persisted jobs from storage
    if settings.JOB_LOAD_ON_STARTUP:
        job_manager = get_job_manager()
        loaded = await job_manager.load_jobs_from_storage()
        if loaded > 0:
            logger.info(f"Loaded {loaded} jobs from storage")

    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="AI text classification service using LLMs with async job management",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(classify.router)
app.include_router(jobs.router)
app.include_router(models.router)


@app.get("/")
async def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "openapi": "/openapi.json"
    }
