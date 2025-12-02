"""cost.py

Cost estimation and calculation for LLM classification jobs.
Uses LiteLLM's dynamic pricing data for accurate, up-to-date cost calculations.
"""

from typing import Dict, Optional
from loguru import logger
from litellm import model_cost, token_counter

from classifier_fastapi.techniques import BaseTechnique
from classifier_fastapi.formatter import formatter

__all__ = [
    "CostEstimator",
]

# Default output token estimate per classification
# Based on real API testing with gpt-4o-mini, gemini-2.5-flash-lite, claude-3-5-haiku:
# Actual output is ~8 tokens per simple zero-shot classification (JSON formatted response)
DEFAULT_OUTPUT_TOKENS_PER_CLASSIFICATION = 8


class CostEstimator:
    """Cost estimation for LLM classification jobs using LiteLLM's dynamic pricing"""

    @staticmethod
    def get_model_pricing(model: str) -> Optional[Dict]:
        """
        Get pricing information for a model from LiteLLM's model_cost dictionary.

        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet-20241022')

        Returns:
            Dict with pricing info, or None if not available
        """
        # Try exact match first
        if model in model_cost:
            pricing = model_cost[model]
            # Only return if it's a chat/completion model
            if pricing.get("mode") in ["chat", "completion"]:
                return pricing

        # Try with provider prefix stripped (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini")
        if "/" in model:
            base_model = model.split("/")[-1]
            if base_model in model_cost:
                pricing = model_cost[base_model]
                if pricing.get("mode") in ["chat", "completion"]:
                    return pricing

        return None

    @staticmethod
    def is_pricing_available(model: str) -> bool:
        """Check if pricing is available for a model"""
        return CostEstimator.get_model_pricing(model) is not None

    @staticmethod
    def estimate_tokens(
        texts: list[str],
        model: str,
        prompt_maker: BaseTechnique,
    ) -> Dict:
        """
        Estimate token usage for a batch of texts using LiteLLM's token_counter.

        Args:
            texts: List of texts to classify
            model: Model name for tokenization
            prompt_maker: Technique to generate prompts

        Returns:
            Dict with keys: input_tokens, estimated_output_tokens, total_tokens, warning (optional)
        """
        try:
            total_input_tokens = 0

            for i, text in enumerate(texts):
                # Generate base prompt
                base_prompt = prompt_maker.make_prompt(text)
                # Add formatting instructions (same as actual classification)
                full_prompt = formatter.format_prompt(
                    prompt=base_prompt,
                    output_keys=prompt_maker.template.output_keys
                )
                # Format as LiteLLM messages
                messages = [{"role": "user", "content": full_prompt}]

                try:
                    tokens = token_counter(model=model, messages=messages)
                    total_input_tokens += tokens

                    # Debug logging for first few texts
                    if i < 3:
                        logger.debug(f"Text {i}: {text[:50]}...")
                        logger.debug(f"  Base prompt length: {len(base_prompt)} chars")
                        logger.debug(f"  Full prompt length: {len(full_prompt)} chars")
                        logger.debug(f"  Tokens counted: {tokens}")
                except Exception as e:
                    logger.warning(f"Failed to count tokens for model {model}: {e}")
                    return {
                        "input_tokens": None,
                        "estimated_output_tokens": None,
                        "total_tokens": None,
                        "warning": f"Token counting not available for {model}"
                    }

            # Estimate output tokens (rough approximation)
            total_output_tokens = len(texts) * DEFAULT_OUTPUT_TOKENS_PER_CLASSIFICATION

            logger.info(f"Cost estimation complete: {len(texts)} texts, {total_input_tokens} input tokens ({total_input_tokens/len(texts):.1f} avg/text)")

            return {
                "input_tokens": total_input_tokens,
                "estimated_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            return {
                "input_tokens": None,
                "estimated_output_tokens": None,
                "total_tokens": None,
                "warning": f"Token estimation failed: {str(e)}"
            }

    @staticmethod
    def estimate_cost(
        tokens: Dict,
        model: str,
    ) -> Optional[float]:
        """
        Estimate cost in USD based on token counts and LiteLLM pricing data.

        Args:
            tokens: Dict with input_tokens and estimated_output_tokens
            model: Model name

        Returns:
            Estimated cost in USD, or None if pricing not available
        """
        if tokens.get("input_tokens") is None:
            return None

        pricing = CostEstimator.get_model_pricing(model)

        if not pricing:
            logger.warning(f"Pricing not available for model: {model}")
            return None

        input_cost_per_token = pricing.get("input_cost_per_token", 0)
        output_cost_per_token = pricing.get("output_cost_per_token", 0)

        if input_cost_per_token == 0 and output_cost_per_token == 0:
            logger.warning(f"Zero pricing found for model: {model}")
            return None

        input_cost = tokens["input_tokens"] * input_cost_per_token
        output_cost = tokens["estimated_output_tokens"] * output_cost_per_token

        return input_cost + output_cost

    @staticmethod
    def calculate_actual_cost(
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> Optional[float]:
        """
        Calculate actual cost based on real token usage.

        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens used
            model: Model name

        Returns:
            Actual cost in USD, or None if pricing not available
        """
        pricing = CostEstimator.get_model_pricing(model)

        if not pricing:
            logger.warning(f"Pricing not available for model: {model}")
            return None

        input_cost_per_token = pricing.get("input_cost_per_token", 0)
        output_cost_per_token = pricing.get("output_cost_per_token", 0)

        if input_cost_per_token == 0 and output_cost_per_token == 0:
            return None

        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token

        return input_cost + output_cost

    @staticmethod
    def get_pricing_per_million(model: str) -> Optional[Dict[str, float]]:
        """
        Get pricing per million tokens for display purposes.

        Args:
            model: Model name

        Returns:
            Dict with input_per_million and output_per_million, or None
        """
        pricing = CostEstimator.get_model_pricing(model)

        if not pricing:
            return None

        input_cost = pricing.get("input_cost_per_token", 0)
        output_cost = pricing.get("output_cost_per_token", 0)

        if input_cost == 0 and output_cost == 0:
            return None

        return {
            "input_per_million": input_cost * 1_000_000,
            "output_per_million": output_cost * 1_000_000
        }
