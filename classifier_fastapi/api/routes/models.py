"""models.py

Model and provider listing endpoints.
"""

from fastapi import APIRouter, Depends
from loguru import logger

from classifier_fastapi.api.models import ModelsListResponse, ModelInfo
from classifier_fastapi.api.dependencies import get_current_api_key
from classifier_fastapi.providers import LLMProvider
from classifier_fastapi.utils.litellm_ import get_available_models
from classifier_fastapi.core.cost import CostEstimator

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/", response_model=ModelsListResponse)
async def list_models(api_key: str = Depends(get_current_api_key)):
    """
    List all available models and providers.

    Uses LiteLLM's model_cost to dynamically discover available models
    and their pricing information.
    """
    models = []
    providers = []

    for provider in LLMProvider:
        providers.append(provider.value)

        try:
            # Get available models for this provider from LiteLLM
            model_names = get_available_models(provider.value)

            for model_name in model_names:
                try:
                    # Get pricing info from LiteLLM
                    pricing = CostEstimator.get_model_pricing(model_name)

                    if pricing:
                        context_window = pricing.get("max_input_tokens")
                        input_cost_per_1m = pricing.get("input_cost_per_token", 0) * 1_000_000
                        output_cost_per_1m = pricing.get("output_cost_per_token", 0) * 1_000_000

                        models.append(ModelInfo(
                            name=model_name,
                            provider=provider.value,
                            context_window=context_window,
                            input_cost_per_1m_tokens=input_cost_per_1m if input_cost_per_1m > 0 else None,
                            output_cost_per_1m_tokens=output_cost_per_1m if output_cost_per_1m > 0 else None
                        ))
                    else:
                        # Include model even if pricing not available
                        models.append(ModelInfo(
                            name=model_name,
                            provider=provider.value,
                            context_window=None,
                            input_cost_per_1m_tokens=None,
                            output_cost_per_1m_tokens=None
                        ))
                except Exception as e:
                    logger.warning(f"Could not get info for model {model_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error listing models for provider {provider.value}: {e}")
            continue

    return ModelsListResponse(models=models, providers=providers)
