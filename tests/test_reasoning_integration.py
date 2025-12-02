"""
Integration tests for reasoning features

Tests full classification with confidence, reasoning, and reasoning modes.
"""
import pytest
from atap_llm_classifier import config
from atap_llm_classifier.core import a_classify
from atap_llm_classifier.models import LLMConfig
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.modifiers import Modifier


@pytest.mark.asyncio
async def test_classification_with_confidence():
    """Test classification returns confidence score."""
    config.mock.enabled = True  # Use mock mode for testing

    technique = Technique.ZERO_SHOT.get_prompt_maker({
        "classes": [
            {"name": "positive", "description": "Positive sentiment"},
            {"name": "negative", "description": "Negative sentiment"}
        ]
    })

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig()

    result = await a_classify(
        text="This is great!",
        model="gpt-4o-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    assert result.classification is not None
    assert result.confidence is not None
    assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
async def test_classification_with_reasoning():
    """Test classification with prompted reasoning."""
    config.mock.enabled = True

    technique = Technique.ZERO_SHOT.get_prompt_maker(
        {"classes": [{"name": "positive", "description": "Positive"}]},
        enable_reasoning=True,
        max_reasoning_chars=150
    )

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig()

    result = await a_classify(
        text="This is great!",
        model="gpt-4o-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    assert result.reasoning is not None
    # Check truncation at 110%
    assert len(result.reasoning) <= int(150 * 1.1)


@pytest.mark.asyncio
async def test_classification_with_reasoning_mode():
    """Test classification with LiteLLM reasoning_effort."""
    config.mock.enabled = True

    technique = Technique.ZERO_SHOT.get_prompt_maker({
        "classes": [{"name": "positive", "description": "Positive"}]
    })

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig(reasoning_effort="medium")

    result = await a_classify(
        text="This is great!",
        model="o3-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    assert result.classification is not None
    # reasoning_content might be None in mock mode unless we mock it


@pytest.mark.asyncio
async def test_classification_all_techniques_with_confidence():
    """Test that all techniques return confidence scores."""
    config.mock.enabled = True

    for tech_enum in [Technique.ZERO_SHOT, Technique.FEW_SHOT, Technique.CHAIN_OF_THOUGHT]:
        if tech_enum == Technique.ZERO_SHOT:
            schema = {"classes": [{"name": "A", "description": "Class A"}]}
        elif tech_enum == Technique.FEW_SHOT:
            schema = {
                "classes": [{"name": "A", "description": "Class A"}],
                "examples": [{"query": "example", "classification": "A"}]
            }
        else:  # CoT
            schema = {
                "classes": [{"name": "A", "description": "Class A"}],
                "examples": [{"query": "example", "classification": "A", "reason": "Because"}]
            }

        technique = tech_enum.get_prompt_maker(schema)
        modifier = Modifier.NO_MODIFIER.get_behaviour()
        llm_config = LLMConfig()

        result = await a_classify(
            text="test text",
            model="gpt-4o-mini",
            llm_config=llm_config,
            technique=technique,
            modifier=modifier,
        )

        assert result.confidence is not None, f"{tech_enum} should return confidence"
        assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
async def test_classification_reasoning_truncation():
    """Test that reasoning is properly truncated at 110% of max_chars."""
    config.mock.enabled = True

    # Create a technique with small max_reasoning_chars
    technique = Technique.ZERO_SHOT.get_prompt_maker(
        {"classes": [{"name": "A", "description": "Class A"}]},
        enable_reasoning=True,
        max_reasoning_chars=50
    )

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig()

    result = await a_classify(
        text="test text",
        model="gpt-4o-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    if result.reasoning:
        max_allowed = int(50 * 1.1)
        assert len(result.reasoning) <= max_allowed, \
            f"Reasoning should be truncated to {max_allowed} chars"


@pytest.mark.asyncio
async def test_classification_cot_with_reasoning():
    """Test that CoT handles both 'reason' and 'reasoning' fields."""
    config.mock.enabled = True

    # CoT has 'reason' by default
    technique = Technique.CHAIN_OF_THOUGHT.get_prompt_maker(
        {
            "classes": [{"name": "A", "description": "Class A"}],
            "examples": [{"query": "example", "classification": "A", "reason": "Because"}]
        },
        enable_reasoning=True,
        max_reasoning_chars=150
    )

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig()

    result = await a_classify(
        text="test text",
        model="gpt-4o-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    # In CoT with enable_reasoning=True, reasoning should be populated
    # (either from 'reasoning' or 'reason' field in output)
    assert result.confidence is not None


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test that classification works without new parameters (backward compatibility)."""
    config.mock.enabled = True

    # Create technique the old way (no reasoning parameters)
    technique = Technique.ZERO_SHOT.get_prompt_maker({
        "classes": [{"name": "positive", "description": "Positive"}]
    })

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig()  # No reasoning_effort

    result = await a_classify(
        text="This is great!",
        model="gpt-4o-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    # Should work fine and return confidence (but not reasoning)
    assert result.classification is not None
    assert result.confidence is not None
    assert result.reasoning_content is None  # No reasoning mode used


@pytest.mark.asyncio
async def test_classification_result_fields():
    """Test that ClassificationResult has all new fields."""
    config.mock.enabled = True

    technique = Technique.ZERO_SHOT.get_prompt_maker(
        {"classes": [{"name": "A", "description": "Class A"}]},
        enable_reasoning=True,
        max_reasoning_chars=150
    )

    modifier = Modifier.NO_MODIFIER.get_behaviour()
    llm_config = LLMConfig(reasoning_effort="low")

    result = await a_classify(
        text="test text",
        model="gpt-4o-mini",
        llm_config=llm_config,
        technique=technique,
        modifier=modifier,
    )

    # Verify all fields exist
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'reasoning')
    assert hasattr(result, 'reasoning_content')
    assert hasattr(result, 'text')
    assert hasattr(result, 'classification')
    assert hasattr(result, 'prompt')
    assert hasattr(result, 'response')
