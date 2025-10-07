"""models.py

Model and provider listing endpoints.
"""

from fastapi import APIRouter, Depends
from loguru import logger

from classifier_fastapi.api.models import ModelsListResponse, ModelInfo
from classifier_fastapi.api.dependencies import get_current_api_key
from classifier_fastapi.providers import LLMProvider

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/", response_model=ModelsListResponse)
async def list_models(api_key: str = Depends(get_current_api_key)):
    """List all available models and providers"""
    models = []
    providers = []

    for provider in LLMProvider:
        providers.append(provider.value)
        # Get available models for this provider
        try:
            props = provider.properties
            # For each provider, we can list common models
            # This is a simplified version - in reality you might query the provider
            if provider.value == "openai":
                model_names = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            elif provider.value == "anthropic":
                model_names = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
            elif provider.value == "ollama":
                model_names = ["llama2", "mistral", "mixtral"]
            else:
                continue

            for model_name in model_names:
                try:
                    model_props = props.get_model_props(model_name)
                    models.append(ModelInfo(
                        name=model_name,
                        provider=provider.value,
                        context_window=model_props.context_window if model_props.known_context_window() else None
                    ))
                except Exception as e:
                    logger.warning(f"Could not get props for model {model_name}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error listing models for provider {provider.value}: {e}")
            continue

    return ModelsListResponse(models=models, providers=providers)
