"""
API Request/Response Models

Pydantic models for FastAPI endpoints.
"""
from pydantic import BaseModel, Field, field_validator
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
    reasoning_effort: Optional[str] = Field(
        None,
        description="Reasoning mode: 'low', 'medium', or 'high'"
    )
    enable_reasoning: bool = Field(
        False,
        description="Enable reasoning output in classification results"
    )
    max_reasoning_chars: int = Field(
        150,
        gt=0,
        description="Maximum characters for reasoning output"
    )
    llm_api_key: Optional[str] = Field(None, description="LLM provider API key")
    llm_endpoint: Optional[str] = Field(None, description="Custom LLM endpoint")

    @field_validator("reasoning_effort", mode="after")
    @classmethod
    def validate_reasoning_effort(cls, v: str | None):
        if v is not None and v not in ["low", "medium", "high"]:
            raise ValueError("reasoning_effort must be 'low', 'medium', 'high', or None")
        return v


class CostEstimateRequest(BaseModel):
    """Request model for cost estimation"""
    texts: List[str] = Field(..., description="Sample texts for cost estimation")
    user_schema: Dict[str, Any] = Field(..., description="Classification schema")
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    technique: str = Field("zero_shot", description="Classification technique")
    enable_reasoning: bool = Field(False, description="Enable reasoning for token estimation")
    max_reasoning_chars: int = Field(150, gt=0)


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
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Reasoning for classification")
    reasoning_content: Optional[str] = Field(None, description="Native reasoning mode output")


class JobStatusResponse(BaseModel):
    """Response for job status queries"""
    job_id: str
    status: JobStatus
    progress: JobProgress
    results: Optional[List[ClassificationResultItem]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    cost: Optional[Dict[str, Any]] = None  # Allow any type for token counts and costs
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
    input_tokens: Optional[int] = Field(None, description="Estimated input tokens")
    output_tokens: Optional[int] = Field(None, description="Estimated output tokens")
    reasoning_tokens: Optional[int] = Field(None, description="Estimated reasoning tokens")
    input_cost_usd: Optional[float] = Field(None, description="Input cost in USD")
    output_cost_usd: Optional[float] = Field(None, description="Output cost in USD")
    reasoning_cost_usd: Optional[float] = Field(None, description="Reasoning cost in USD")
    input_cost_per_1m: Optional[float] = Field(None, description="Input cost per 1M tokens (USD)")
    output_cost_per_1m: Optional[float] = Field(None, description="Output cost per 1M tokens (USD)")
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
    input_cost_per_1m_tokens: Optional[float] = Field(None, description="Input cost per 1M tokens (USD)")
    output_cost_per_1m_tokens: Optional[float] = Field(None, description="Output cost per 1M tokens (USD)")


class ModelsListResponse(BaseModel):
    """Response for model listing"""
    models: List[ModelInfo]
    providers: List[str]
