# Classifier FastAPI

AI text classification service using LLMs with async job management and parallel processing.

## Overview

This FastAPI service provides a REST API for batch text classification using various LLM providers. It uses an async worker pool architecture to process multiple classification requests concurrently while respecting API rate limits.

## Directory Structure

```
classifier_fastapi/
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── models.py            # Pydantic request/response models
│   ├── dependencies.py      # API key validation
│   └── routes/
│       ├── classify.py      # Batch classification endpoints
│       ├── jobs.py          # Job status endpoints
│       ├── models.py        # Model listing endpoint
│       └── health.py        # Health check endpoints
├── core/
│   ├── core.py              # Single text classification (a_classify)
│   ├── pipeline.py          # Batch processing with worker pool
│   ├── cost.py              # Cost estimation using LiteLLM pricing
│   ├── config.py            # Core configuration
│   └── models.py            # LLM config models
├── providers/               # LLM provider abstractions (OpenAI, Ollama, etc.)
├── techniques/              # Classification techniques (zero-shot, chain-of-thought)
├── modifiers/               # Response modifiers (self-consistency, etc.)
├── formatter/               # Output format handling (JSON/YAML)
├── job_manager.py           # Job tracking with optional persistence
├── storage.py               # JSON file-based job storage
├── ratelimiters.py          # Token bucket rate limiting
└── settings.py              # Environment-based settings
```

## API Endpoints

### Classification

#### `POST /classify/batch`
Submit a batch classification job.

**Request Body:**
```json
{
  "texts": ["text1", "text2", "text3"],
  "user_schema": {
    "classes": [
      {"name": "positive", "description": "Positive sentiment"},
      {"name": "negative", "description": "Negative sentiment"}
    ]
  },
  "provider": "openai",
  "model": "gpt-4o-mini",
  "technique": "zero_shot",
  "modifier": "no_modifier",
  "temperature": 1.0,
  "top_p": 1.0,
  "reasoning_effort": null,
  "enable_reasoning": false,
  "max_reasoning_chars": 150,
  "llm_api_key": "sk-...",
  "llm_endpoint": null
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Classification job created successfully",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### `POST /classify/estimate-cost`
Estimate classification cost before running. Use this to preview costs without consuming API credits.

**Request Body:**
```json
{
  "texts": ["sample text 1", "sample text 2"],
  "user_schema": {
    "classes": [
      {"name": "positive", "description": "Positive sentiment"},
      {"name": "negative", "description": "Negative sentiment"}
    ]
  },
  "provider": "openai",
  "model": "gpt-4o-mini",
  "technique": "zero_shot",
  "enable_reasoning": false,
  "max_reasoning_chars": 150
}
```

**Response:**
```json
{
  "estimated_tokens": 15000,
  "estimated_cost_usd": 0.0045,
  "provider": "openai",
  "model": "gpt-4o-mini",
  "num_texts": 100,
  "input_tokens": 12000,
  "output_tokens": 3000,
  "reasoning_tokens": 0,
  "input_cost_usd": 0.0018,
  "output_cost_usd": 0.0018,
  "reasoning_cost_usd": 0.0,
  "input_cost_per_1m": 0.15,
  "output_cost_per_1m": 0.60,
  "warnings": []
}
```

**Cost Calculation:**
- Uses LiteLLM's built-in pricing database for accurate per-model costs
- Estimates input tokens by tokenizing the full prompt (including schema)
- Estimates output tokens based on expected response structure
- Reasoning tokens calculated separately when `reasoning_effort` is set

### Job Management

#### `GET /jobs/`
List all jobs with optional filtering.

**Query Parameters:**
- `status` (optional): Filter by status (`pending`, `running`, `completed`, `failed`, `cancelled`)
- `limit` (optional): Maximum jobs to return (default: 100)

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress": {
      "total": 100,
      "completed": 100,
      "failed": 0,
      "percentage": 100.0
    },
    "created_at": "2024-01-15T10:30:00",
    "started_at": "2024-01-15T10:30:01",
    "completed_at": "2024-01-15T10:35:00"
  }
]
```

#### `GET /jobs/{job_id}`
Get job status and results.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": {
    "total": 100,
    "completed": 98,
    "failed": 2,
    "percentage": 100.0
  },
  "results": [
    {
      "index": 0,
      "text": "original text",
      "classification": "positive",
      "confidence": 0.95,
      "reasoning": "The text expresses satisfaction..."
    }
  ],
  "errors": [
    {"index": 5, "error": "Rate limit exceeded"}
  ],
  "cost": {
    "total_usd": 0.0042,
    "total_tokens": 14500,
    "input_tokens": 11800,
    "output_tokens": 2700
  }
}
```

#### `DELETE /jobs/{job_id}`
Cancel a running or pending job.

### Models

#### `GET /models/`
List available models with pricing information.

### Health

#### `GET /health/`
Health check with uptime.

#### `GET /health/ready`
Readiness probe.

## Batch Processing Architecture

### How It Works

The batch classification uses a **parallel worker pool pattern**, not true LLM batch inference:

```
POST /classify/batch
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Job Created (UUID, status=PENDING)                     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Background Task: process_classification_job()          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  asyncio.Queue                                   │   │
│  │  [text1, text2, text3, ..., textN]              │   │
│  └─────────────────────────────────────────────────┘   │
│           │         │         │         │               │
│           ▼         ▼         ▼         ▼               │
│      ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│      │Worker 1│ │Worker 2│ │Worker 3│ │Worker N│       │
│      └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘       │
│           │         │         │         │               │
│           ▼         ▼         ▼         ▼               │
│      a_classify  a_classify  a_classify  a_classify     │
│      (1 LLM call per text)                              │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Results Aggregated → Cost Calculated → Job COMPLETED   │
└─────────────────────────────────────────────────────────┘
```

### Key Points

- **One text = One LLM API call**: Each text is classified individually
- **Parallelism via workers**: Multiple async workers process texts concurrently
- **Rate limiting**: Token bucket algorithm prevents API rate limit errors
- **Progress tracking**: Real-time progress updates via job polling

### Configuration

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| Max batch size | `MAX_BATCH_SIZE` | 1000 | Maximum texts per job |
| Workers | `DEFAULT_WORKERS` | 5 | Concurrent LLM API calls |
| Max concurrent jobs | `MAX_CONCURRENT_JOBS` | 100 | Jobs in memory |
| Job timeout | `JOB_TIMEOUT_SECONDS` | 3600 | Max job duration |

### Job Persistence

Jobs are persisted to disk as JSON files, ensuring they survive server restarts. Results are stored incrementally to prevent data loss on crashes.

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| Storage directory | `JOB_STORAGE_DIR` | `data/jobs` | Directory for job JSON files |
| Enable persistence | `JOB_PERSIST_ENABLED` | `true` | Enable/disable persistence |
| Load on startup | `JOB_LOAD_ON_STARTUP` | `true` | Load existing jobs on startup |
| Batch size | `JOB_PERSIST_BATCH_SIZE` | 50 | Persist every N results |
| Memory threshold | `JOB_MEMORY_THRESHOLD_MB` | 100 | Switch to disk-only mode above this |

**Features:**
- **Atomic writes**: Uses temp file + rename to prevent corruption
- **Batched persistence**: Results are persisted every 50 items (configurable) to reduce I/O
- **Memory management**: Jobs exceeding 100MB switch to disk-only mode
- **Graceful restart**: Running jobs are marked as failed on server restart
- **Fault-tolerant**: Storage failures log warnings but don't break processing

### Rate Limiting

The service implements a **token bucket** rate limiter with:

- **Request rate limiting**: Controls requests per time window
- **Token rate limiting**: Controls tokens per minute (provider-specific)
- **Exponential backoff**: Automatic retry on rate limit errors
- **Provider-specific limits**: Queries OpenAI headers for actual limits

## Request Parameters

### ClassificationRequest

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `texts` | `List[str]` | Yes | - | Texts to classify |
| `user_schema` | `Dict` | Yes | - | Classification schema with classes |
| `provider` | `str` | Yes | - | LLM provider: `openai`, `ollama`, `anthropic`, etc. |
| `model` | `str` | Yes | - | Model name (e.g., `gpt-4o-mini`) |
| `technique` | `str` | No | `zero_shot` | Classification technique |
| `modifier` | `str` | No | `no_modifier` | Response modifier |
| `temperature` | `float` | No | `1.0` | Sampling temperature (0-2) |
| `top_p` | `float` | No | `1.0` | Nucleus sampling (0-1) |
| `reasoning_effort` | `str` | No | `null` | `low`, `medium`, or `high` |
| `enable_reasoning` | `bool` | No | `false` | Include reasoning in output |
| `max_reasoning_chars` | `int` | No | `150` | Max reasoning length |
| `llm_api_key` | `str` | No | - | Provider API key |
| `llm_endpoint` | `str` | No | - | Custom endpoint URL |

### User Schema Format

```json
{
  "classes": [
    {"name": "class_name", "description": "Description of the class"}
  ]
}
```

## Authentication

The API uses header-based authentication:

```bash
curl -X POST /classify/batch \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"texts": [...], ...}'
```

Configure API keys via environment variables:
- `SERVICE_API_KEY`: Single API key
- `SERVICE_API_KEYS`: Comma-separated multiple keys

## Running the Service

### Development

```bash
# Using run_api.py
python run_api.py

# Using uvicorn directly
uvicorn classifier_fastapi.api.main:app --host 0.0.0.0 --port 8002 --reload
```

### API Documentation

Once the service is running, access the interactive API documentation:

| URL | Description |
|-----|-------------|
| http://localhost:8002/docs | **Swagger UI** - Interactive API explorer with "Try it out" functionality |
| http://localhost:8002/redoc | **ReDoc** - Clean, readable API documentation |
| http://localhost:8002/openapi.json | **OpenAPI Schema** - Raw JSON schema for code generation |

The Swagger UI (`/docs`) allows you to:
- Browse all endpoints and their parameters
- Test API calls directly from the browser
- View request/response schemas
- Authenticate using the "Authorize" button (enter your API key)

### Environment Variables

```bash
# .env file
SERVICE_API_KEY=your-api-key
MAX_BATCH_SIZE=1000
DEFAULT_WORKERS=5
CORS_ORIGINS=*
LOG_LEVEL=INFO
LLM_OUTPUT_FORMAT=json
OLLAMA_ENDPOINT=http://127.0.0.1:11434

# Job persistence
JOB_STORAGE_DIR=data/jobs
JOB_PERSIST_ENABLED=true
JOB_LOAD_ON_STARTUP=true
JOB_PERSIST_BATCH_SIZE=50
JOB_MEMORY_THRESHOLD_MB=100
```

## Example Usage

### Submit a Batch Job

```bash
curl -X POST http://localhost:8002/classify/batch \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["I love this product!", "This is terrible."],
    "user_schema": {
      "classes": [
        {"name": "positive", "description": "Positive sentiment"},
        {"name": "negative", "description": "Negative sentiment"}
      ]
    },
    "provider": "openai",
    "model": "gpt-4o-mini",
    "llm_api_key": "sk-..."
  }'
```

### Poll for Results

```bash
curl http://localhost:8002/jobs/{job_id} \
  -H "X-API-Key: your-api-key"
```

### Estimate Cost Before Running

```bash
curl -X POST http://localhost:8002/classify/estimate-cost \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Sample text to estimate cost"],
    "user_schema": {
      "classes": [
        {"name": "positive", "description": "Positive sentiment"},
        {"name": "negative", "description": "Negative sentiment"}
      ]
    },
    "provider": "openai",
    "model": "gpt-4o-mini"
  }'
```

## Job States

| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting to start |
| `running` | Classification in progress |
| `completed` | All texts processed |
| `failed` | Job failed with error |
| `cancelled` | Job cancelled by user |

## Error Handling

- **Rate limit errors**: Automatic retry with exponential backoff
- **Unsupported parameters**: Automatic retry without unsupported params (e.g., `reasoning_effort`)
- **Context window exceeded**: Pre-flight validation fails the job
- **Invalid schema**: Returns 400 with validation error
