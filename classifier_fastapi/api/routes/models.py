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
    """
    List all available models and providers.

    For Ollama: queries the actual local Ollama instance.
    For other providers: uses LiteLLM's model_cost database.
    """
    models = []
    providers = []

    for provider in LLMProvider:
        providers.append(provider.value)

        try:
            # Get provider properties which handles Ollama differently
            provider_props = provider.properties

            # Use models from provider properties
            for model_prop in provider_props.models:
                try:
                    # Convert token costs to per-million format
                    input_cost_per_1m = None
                    output_cost_per_1m = None

                    if model_prop.input_token_cost is not None:
                        input_cost_per_1m = model_prop.input_token_cost * 1_000_000
                    if model_prop.output_token_cost is not None:
                        output_cost_per_1m = model_prop.output_token_cost * 1_000_000

                    models.append(ModelInfo(
                        name=model_prop.name,
                        provider=provider.value,
                        context_window=model_prop.context_window,
                        input_cost_per_1m_tokens=input_cost_per_1m if input_cost_per_1m and input_cost_per_1m > 0 else None,
                        output_cost_per_1m_tokens=output_cost_per_1m if output_cost_per_1m and output_cost_per_1m > 0 else None
                    ))
                except Exception as e:
                    logger.warning(f"Could not get info for model {model_prop.name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error listing models for provider {provider.value}: {e}")
            continue

    return ModelsListResponse(models=models, providers=providers)
