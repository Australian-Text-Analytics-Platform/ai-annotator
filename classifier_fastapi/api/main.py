"""
FastAPI Main Application

Entry point for the classification API service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

# TODO: Import routers when implemented
# from classifier_fastapi.api.routes import health, classify, jobs, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.start_time = time.time()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="ATAP LLM Classifier API",
    version="0.1.0",
    description="AI text classification service using LLMs",
    lifespan=lifespan
)

# CORS middleware - will be configured from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure from settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Include routers
# app.include_router(health.router)
# app.include_router(classify.router)
# app.include_router(jobs.router)
# app.include_router(models.router)


@app.get("/")
async def root():
    return {
        "service": "ATAP LLM Classifier API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }
