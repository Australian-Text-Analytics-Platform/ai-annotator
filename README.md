# AI Annotator

This repository provides an AI annotator tool for the LDaCA/ATAP Platform.
The tool enables automated text classification using large language models with support for zero-shot, few-shot, and chain-of-thought prompting techniques.

Available as both a **CLI tool** and **REST API service** for flexible integration.


## Installation 

```shell
python3.11 -m venv .venv
source .venv/bin/activate

# pipx install poetry
poetry install

atapllmc classify batch --help
```

## Quick Start

### 1. Create a User Schema

Create a JSON file defining your classification classes:

```json
{
  "classes": [
    {
      "name": "positive",
      "description": "Positive sentiment or opinion"
    },
    {
      "name": "negative",
      "description": "Negative sentiment or opinion"
    }
  ]
}
```

### 2. Run Classification

**Using OpenAI:**
```shell
atapllmc classify batch \
  --dataset 'example.csv' \
  --column 'text' \
  --out-dir './out' \
  --provider openai \
  --model 'gpt-4.1-mini' \
  --technique zero_shot \
  --user-schema 'user_schema.json' \
  --api-key <your-api-key>
```

**Using Ollama (local):**
```shell
atapllmc classify batch \
  --dataset 'example.csv' \
  --column 'text' \
  --out-dir './out' \
  --provider ollama \
  --model 'llama3:8b' \
  --technique zero_shot \
  --user-schema 'user_schema.json'
  # --endpoint <custom-endpoint>  # Optional: default is http://127.0.0.1:11434
```

## Python API Usage

You can use the AI Annotator pipeline directly in Python code:

```python
import pandas as pd
from atap_corpus import Corpus
from atap_llm_classifier import pipeline
from atap_llm_classifier.models import LLMConfig
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.providers import LLMProvider
from atap_llm_classifier.techniques import Technique

# Load your data
df = pd.read_csv('your_data.csv')
corpus = Corpus.from_dataframe(df=df, col_doc='text')

# Define user schema
user_schema = {
    "classes": [
        {"name": "positive", "description": "Positive sentiment"},
        {"name": "negative", "description": "Negative sentiment"}
    ]
}

# Configure model
provider = LLMProvider.OPENAI
model_props = provider.properties.with_api_key("your-api-key").get_model_props("gpt-4.1-mini")
llm_config = LLMConfig(temperature=0.7, top_p=0.9)

# Run classification
results = pipeline.batch(
    corpus=corpus,
    model_props=model_props,
    llm_config=llm_config,
    technique=Technique.ZERO_SHOT,  # or FEW_SHOT, CHAIN_OF_THOUGHT
    user_schema=user_schema,
    modifier=Modifier.NO_MODIFIER,
    on_result_callback=lambda result: print(f"Classified: {result.doc_idx}")
)

# Access results
print(f"Successful classifications: {len(results.successes)}")
print(f"Failed classifications: {len(results.fails)}")

for success in results.successes:
    doc_idx = success.doc_idx
    classification = success.classification_result.classification
    print(f"Document {doc_idx}: {classification}")
```

### Using Few-Shot with Examples

```python
# Few-shot user schema with examples
user_schema = {
    "classes": [
        {"name": "positive", "description": "Positive sentiment"},
        {"name": "negative", "description": "Negative sentiment"}
    ],
    "examples": [
        {"query": "I love this!", "classification": "positive"},
        {"query": "This is terrible.", "classification": "negative"}
    ]
}

# Use few-shot technique
results = pipeline.batch(
    corpus=corpus,
    model_props=model_props,
    llm_config=llm_config,
    technique=Technique.FEW_SHOT,
    user_schema=user_schema,
    modifier=Modifier.NO_MODIFIER
)
```

## FastAPI REST API

The classifier is also available as a REST API service for integration with other applications and services.

### Starting the API Server

```shell
# Using the development server
python run_api.py

# Or using uvicorn directly
uvicorn classifier_fastapi.api.main:app --host 0.0.0.0 --port 8002 --reload
```

The API will be available at `http://localhost:8002` with interactive documentation at:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

### Configuration

Set environment variables in `.env` file:

```shell
SERVICE_API_KEYS=your-api-key-1,your-api-key-2
MAX_BATCH_SIZE=1000
MAX_CONCURRENT_JOBS=100
DEFAULT_WORKERS=5
CORS_ORIGINS="*"
```

### API Endpoints

**Authentication**: All endpoints (except `/health/`) require an `X-API-Key` header.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/` | GET | Health check (no auth required) |
| `/models/` | GET | List available models and pricing |
| `/classify/estimate-cost` | POST | Estimate cost before classification |
| `/classify/batch` | POST | Submit classification job |
| `/jobs/{job_id}` | GET | Get job status and results |
| `/jobs/{job_id}` | DELETE | Cancel a running job |

### Example Usage

**Submit a Classification Job:**

```bash
curl -X POST http://localhost:8002/classify/batch \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is great!", "I hate this", "It is okay"],
    "user_schema": {
      "classes": [
        {"name": "positive", "description": "Positive sentiment"},
        {"name": "negative", "description": "Negative sentiment"},
        {"name": "neutral", "description": "Neutral sentiment"}
      ]
    },
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "technique": "zero_shot",
    "temperature": 0.7,
    "llm_api_key": "your-key"
  }'
```

Response:
```json
{
  "job_id": "4e5b8835-34cf-46f7-81d7-c836053ca24d",
  "status": "pending",
  "message": "Classification job created successfully",
  "created_at": "2025-11-17T22:55:26.047218"
}
```

**Check Job Status:**

```bash
curl http://localhost:8002/jobs/4e5b8835-34cf-46f7-81d7-c836053ca24d \
  -H "X-API-Key: test-api-key"
```

Response:
```json
{
  "job_id": "4e5b8835-34cf-46f7-81d7-c836053ca24d",
  "status": "completed",
  "progress": {
    "total": 3,
    "completed": 3,
    "failed": 0,
    "percentage": 100.0
  },
  "results": [
    {
      "index": 0,
      "text": "This is great!",
      "classification": "positive"
    }
  ],
  "cost": {
    "total_usd": 0.000022,
    "total_tokens": 90
  }
}
```

### Python Client Example

```python
import requests

API_URL = "http://localhost:8002"
API_KEY = "your-api-key"

# Submit job
response = requests.post(
    f"{API_URL}/classify/batch",
    headers={"X-API-Key": API_KEY},
    json={
        "texts": ["Amazing product!", "Terrible service"],
        "user_schema": {
            "classes": [
                {"name": "positive", "description": "Positive sentiment"},
                {"name": "negative", "description": "Negative sentiment"}
            ]
        },
        "provider": "openai",
        "model": "gpt-4o-mini",
        "technique": "zero_shot",
        "llm_api_key": "your-openai-key"
    }
)

job_id = response.json()["job_id"]

# Check status
status = requests.get(
    f"{API_URL}/jobs/{job_id}",
    headers={"X-API-Key": API_KEY}
).json()

print(f"Status: {status['status']}")
print(f"Results: {status['results']}")
```

## CLI Reference

### Commands

The main entry point is `atapllmc` with the following subcommands:

- `atapllmc classify batch` - Perform batch text classification
- `atapllmc litellm list-models` - List available LLM models

### Batch Classification Arguments

#### Required Arguments

- `--dataset TEXT` - Path to input dataset (CSV or XLSX file) **[required]**
- `--column TEXT` - Column name containing text to classify **[required]**
- `--out-dir TEXT` - Output directory for results **[required]**
- `--provider [openai|openai_azure_sih|ollama]` - LLM provider **[required]**
- `--model TEXT` - Model name to use for classification **[required]**
- `--user-schema TEXT` - Path to JSON schema file or raw JSON string **[required]**

#### Optional Arguments

- `--technique [zero_shot|few_shot|chain_of_thought]` - Classification technique (default: `zero_shot`)
- `--modifier [no_modifier|self_consistency]` - Response modification behavior (default: `no_modifier`)
- `--temperature FLOAT` - Model temperature parameter (0.0-1.0)
- `--top-p FLOAT` - Model top-p parameter (0.0-1.0)
- `--api-key TEXT` - API key for the provider (if required)
- `--endpoint TEXT` - Custom endpoint URL (if different from default)
- `--help` - Show help message and exit

### Providers

Available providers and their supported models:

#### OpenAI (`openai`)
- Models: `gpt-3.5-turbo`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4-turbo`, etc.
- Requires: `--api-key`
- Example: `--provider openai --model gpt-4.1-mini --api-key <your-key>`

#### OpenAI Azure SIH (`openai_azure_sih`)
- Azure-hosted OpenAI models via Sydney Infrastructure Hub
- Requires: `--api-key`
- Example: `--provider openai_azure_sih --model gpt-4.1-mini --api-key <your-key>`

#### Ollama (`ollama`)
- Models: `llama3:8b`, `llama3:70b`, `mistral`, etc. (depends on local installation)
- Default endpoint: `http://127.0.0.1:11434`
- Example: `--provider ollama --model llama3:8b`
- Custom endpoint: `--provider ollama --model llama3:8b --endpoint http://custom-url:11434`

### Classification Techniques

#### Zero Shot (`zero_shot`)
- Default technique
- No examples required, classifies based on class descriptions only
- User schema format:
```json
{
  "classes": [
    {"name": "positive", "description": "Positive sentiment text"},
    {"name": "negative", "description": "Negative sentiment text"}
  ]
}
```

#### Few Shot (`few_shot`)
- Provides examples without reasoning to guide the LLM
- Uses query-classification pairs to demonstrate the task
- User schema format:
```json
{
  "classes": [
    {"name": "positive", "description": "Positive sentiment"},
    {"name": "negative", "description": "Negative sentiment"}
  ],
  "examples": [
    {
      "query": "I love this product!",
      "classification": "positive"
    },
    {
      "query": "This is terrible quality.",
      "classification": "negative"
    }
  ]
}
```

#### Chain of Thought (`chain_of_thought`)
- Uses intermediate reasoning steps
- Requires examples with explicit reasoning
- User schema format:
```json
{
  "classes": [
    {"name": "positive", "description": "Positive sentiment"},
    {"name": "negative", "description": "Negative sentiment"}
  ],
  "examples": [
    {
      "query": "I love this product!",
      "classification": "positive",
      "reason": "The word 'love' indicates strong positive emotion"
    }
  ]
}
```

### Response Modifiers

#### No Modifier (`no_modifier`)
- Default behavior
- Single response per classification

#### Self Consistency (`self_consistency`)
- Generates multiple responses and uses majority vote
- More robust but slower and more expensive

### User Schema Examples

#### Simple Sentiment Analysis
```json
{
  "classes": [
    {"name": "positive", "description": "Positive sentiment or opinion"},
    {"name": "negative", "description": "Negative sentiment or opinion"},
    {"name": "neutral", "description": "Neutral or mixed sentiment"}
  ]
}
```

#### Topic Classification
```json
{
  "classes": [
    {"name": "technology", "description": "Technology, software, or IT related content"},
    {"name": "politics", "description": "Political discussions, policies, or government"},
    {"name": "sports", "description": "Sports, games, or athletic activities"},
    {"name": "entertainment", "description": "Movies, music, TV, or celebrity news"}
  ]
}
```

#### Few Shot Example
```json
{
  "classes": [
    {"name": "urgent", "description": "Requires immediate attention"},
    {"name": "normal", "description": "Standard priority"},
    {"name": "low", "description": "Can be handled later"}
  ],
  "examples": [
    {
      "query": "Server is down, customers can't access the site!",
      "classification": "urgent"
    },
    {
      "query": "Update the documentation when you have time",
      "classification": "low"
    },
    {
      "query": "Review the quarterly reports by next week",
      "classification": "normal"
    }
  ]
}
```

#### Chain of Thought Example
```json
{
  "classes": [
    {"name": "urgent", "description": "Requires immediate attention"},
    {"name": "normal", "description": "Standard priority"},
    {"name": "low", "description": "Can be handled later"}
  ],
  "examples": [
    {
      "query": "Server is down, customers can't access the site!",
      "classification": "urgent",
      "reason": "Server outage affects customer access and business operations"
    },
    {
      "query": "Update the documentation when you have time",
      "classification": "low",
      "reason": "Documentation update is important but not time-sensitive"
    }
  ]
}
```

### Output Files

After classification, the following files are generated in the output directory:

- `results.json` - Classification results and metadata
- `user_schema.json` - User schema used for classification
- `corpus.zip` - Serialized corpus data
- `corpus.csv` - CSV format of the corpus with results