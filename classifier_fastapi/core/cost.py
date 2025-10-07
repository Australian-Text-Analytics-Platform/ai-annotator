"""cost.py

Cost estimation and calculation for LLM classification jobs.
Provides token counting and pricing calculations for different providers.
"""

from typing import Dict, Optional
from loguru import logger

from classifier_fastapi.providers.providers import LLMModelProperties
from classifier_fastapi.techniques import BaseTechnique

# Pricing data (USD per token) - Updated as of 2024
# Source: OpenAI pricing page
PRICING = {
    "openai": {
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
        "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "gpt-4o-2024-08-06": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        "gpt-4-turbo-2024-04-09": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        "gpt-4": {"input": 30.00 / 1_000_000, "output": 60.00 / 1_000_000},
        "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
        "gpt-3.5-turbo-0125": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-5-sonnet-20240620": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-opus-20240229": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
        "claude-3-sonnet-20240229": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-haiku-20240307": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    },
    # Ollama models are typically free (self-hosted)
    "ollama": {},
}

# Default output token estimate per classification
DEFAULT_OUTPUT_TOKENS_PER_CLASSIFICATION = 50


class CostEstimator:
    """Cost estimation for LLM classification jobs"""

    @staticmethod
    def estimate_tokens(
        texts: list[str],
        model_props: LLMModelProperties,
        prompt_maker: BaseTechnique,
    ) -> Dict:
        """
        Estimate token usage for a batch of texts.

        Returns:
            Dict with keys: input_tokens, estimated_output_tokens, total_tokens, warning (optional)
        """
        if not model_props.known_tokeniser():
            logger.warning(f"Tokenizer not available for {model_props.name}")
            return {
                "input_tokens": None,
                "estimated_output_tokens": None,
                "total_tokens": None,
                "warning": f"Tokenizer not available for {model_props.name}"
            }

        total_input_tokens = 0
        for text in texts:
            prompt = prompt_maker.make_prompt(text)
            tokens = model_props.count_tokens(prompt)
            total_input_tokens += tokens

        # Estimate output tokens (rough approximation)
        total_output_tokens = len(texts) * DEFAULT_OUTPUT_TOKENS_PER_CLASSIFICATION

        return {
            "input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        }

    @staticmethod
    def estimate_cost(
        tokens: Dict,
        provider: str,
        model: str,
    ) -> Optional[float]:
        """
        Estimate cost in USD based on token counts and provider pricing.

        Args:
            tokens: Dict with input_tokens and estimated_output_tokens
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name

        Returns:
            Estimated cost in USD, or None if pricing not available
        """
        if tokens.get("input_tokens") is None:
            return None

        provider_lower = provider.lower()

        # Handle litellm model naming (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini")
        model_name = model.split("/")[-1] if "/" in model else model

        if provider_lower not in PRICING:
            logger.warning(f"Pricing not available for provider: {provider}")
            return None

        if model_name not in PRICING[provider_lower]:
            logger.warning(f"Pricing not available for model: {model_name} from provider: {provider}")
            return None

        pricing = PRICING[provider_lower][model_name]
        input_cost = tokens["input_tokens"] * pricing["input"]
        output_cost = tokens["estimated_output_tokens"] * pricing["output"]

        return input_cost + output_cost

    @staticmethod
    def calculate_actual_cost(
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> Optional[float]:
        """
        Calculate actual cost based on real token usage.

        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens used
            provider: Provider name
            model: Model name

        Returns:
            Actual cost in USD, or None if pricing not available
        """
        provider_lower = provider.lower()

        # Handle litellm model naming
        model_name = model.split("/")[-1] if "/" in model else model

        if provider_lower not in PRICING:
            logger.warning(f"Pricing not available for provider: {provider}")
            return None

        if model_name not in PRICING[provider_lower]:
            logger.warning(f"Pricing not available for model: {model_name} from provider: {provider}")
            return None

        pricing = PRICING[provider_lower][model_name]
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]

        return input_cost + output_cost

    @staticmethod
    def get_available_pricing() -> Dict:
        """Get all available pricing information"""
        return PRICING.copy()

    @staticmethod
    def supports_pricing(provider: str, model: str) -> bool:
        """Check if pricing is available for a provider/model combination"""
        provider_lower = provider.lower()
        model_name = model.split("/")[-1] if "/" in model else model

        return (
            provider_lower in PRICING and
            model_name in PRICING[provider_lower]
        )
