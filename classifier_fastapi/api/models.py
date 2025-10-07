"""
API Request/Response Models

Pydantic models for FastAPI endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Request Models
class ClassificationRequest(BaseModel):
    """Request model for batch classification"""
    texts: List[str] = Field(..., description="List of texts to classify")
    user_schema: Dict[str, Any] = Field(..., description="Classification schema")
    provider: str = Field(..., description="LLM provider (openai, ollama, etc.)")
    model: str = Field(..., description="Model name")
    technique: str = Field("zero_shot", description="Classification technique")
    modifier: str = Field("no_modifier", description="Response modifier")
    temperature: Optional[float] = Field(None, gt=0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0, le=1.0)
    llm_api_key: Optional[str] = Field(None, description="LLM provider API key")
    llm_endpoint: Optional[str] = Field(None, description="Custom LLM endpoint")


class CostEstimateRequest(BaseModel):
    """Request model for cost estimation"""
    texts: List[str] = Field(..., description="Sample texts for cost estimation")
    user_schema: Dict[str, Any] = Field(..., description="Classification schema")
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    technique: str = Field("zero_shot", description="Classification technique")


# Response Models
class JobCreateResponse(BaseModel):
    """Response when creating a new job"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus
    message: str = Field(..., description="Human-readable message")
    created_at: str


class JobProgress(BaseModel):
    """Job progress information"""
    total: int
    completed: int
    failed: int
    percentage: float


class ClassificationResultItem(BaseModel):
    """Single classification result"""
    index: int
    text: str
    classification: str
    prompt: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response for job status queries"""
    job_id: str
    status: JobStatus
    progress: JobProgress
    results: Optional[List[ClassificationResultItem]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    cost: Optional[Dict[str, float]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class CostEstimateResponse(BaseModel):
    """Response for cost estimation"""
    estimated_tokens: int
    estimated_cost_usd: Optional[float] = None
    provider: str
    model: str
    num_texts: int
    warnings: List[str] = []


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    provider: str
    context_window: Optional[int] = None
    pricing: Optional[Dict[str, float]] = None


class ModelsListResponse(BaseModel):
    """Response for model listing"""
    models: List[ModelInfo]
    providers: List[str]
