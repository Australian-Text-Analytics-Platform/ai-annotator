"""
Unit tests for reasoning features

Tests model validation, technique reasoning instructions, and output keys.
"""
import pytest
from atap_llm_classifier.models import LLMConfig
from atap_llm_classifier.techniques.zeroshot import ZeroShot
from atap_llm_classifier.techniques.fewshot import FewShot
from atap_llm_classifier.techniques.cot import ChainOfThought


def test_llm_config_reasoning_effort_validation():
    """Test reasoning_effort validation accepts valid values."""
    # Valid values
    config = LLMConfig(reasoning_effort="low")
    assert config.reasoning_effort == "low"

    config = LLMConfig(reasoning_effort="medium")
    assert config.reasoning_effort == "medium"

    config = LLMConfig(reasoning_effort="high")
    assert config.reasoning_effort == "high"

    config = LLMConfig(reasoning_effort=None)
    assert config.reasoning_effort is None

    # Invalid value
    with pytest.raises(ValueError, match="reasoning_effort must be"):
        LLMConfig(reasoning_effort="invalid")


def test_llm_config_claude_top_p_handling():
    """Test that LiteLLMCompletionArgs handles Claude top_p correctly."""
    from atap_llm_classifier.models import LiteLLMCompletionArgs, LiteLLMMessage, LiteLLMRole

    # Test Claude model
    args = LiteLLMCompletionArgs(
        model="claude-3-5-sonnet-20241022",
        messages=[LiteLLMMessage(content="test", role=LiteLLMRole.USER)],
        temperature=1.0,
        top_p=1.0,
        n=1,
    )
    kwargs = args.to_kwargs()
    assert "top_p" not in kwargs, "top_p should be removed for Claude models"

    # Test non-Claude model
    args = LiteLLMCompletionArgs(
        model="gpt-4o-mini",
        messages=[LiteLLMMessage(content="test", role=LiteLLMRole.USER)],
        temperature=1.0,
        top_p=1.0,
        n=1,
    )
    kwargs = args.to_kwargs()
    assert "top_p" in kwargs, "top_p should be preserved for non-Claude models"


def test_technique_reasoning_instruction_zeroshot():
    """Test that ZeroShot generates correct reasoning instructions."""
    schema = {"classes": [{"name": "A", "description": "Class A"}]}

    # Without reasoning
    tech = ZeroShot(schema, enable_reasoning=False)
    prompt = tech.make_prompt("test text")
    assert "explanation for your classification" not in prompt.lower()

    # With reasoning
    tech = ZeroShot(schema, enable_reasoning=True, max_reasoning_chars=100)
    prompt = tech.make_prompt("test text")
    assert "explanation for your classification" in prompt.lower()
    assert "100 characters" in prompt


def test_technique_reasoning_instruction_fewshot():
    """Test that FewShot generates correct reasoning instructions."""
    schema = {
        "classes": [{"name": "A", "description": "Class A"}],
        "examples": [{"query": "example", "classification": "A"}]
    }

    # Without reasoning
    tech = FewShot(schema, enable_reasoning=False)
    prompt = tech.make_prompt("test text")
    assert "explanation for your classification" not in prompt.lower()

    # With reasoning
    tech = FewShot(schema, enable_reasoning=True, max_reasoning_chars=200)
    prompt = tech.make_prompt("test text")
    assert "explanation for your classification" in prompt.lower()
    assert "200 characters" in prompt


def test_technique_reasoning_instruction_cot():
    """Test that ChainOfThought generates correct reasoning instructions."""
    schema = {
        "classes": [{"name": "A", "description": "Class A"}],
        "examples": [{"query": "example", "classification": "A", "reason": "Because"}]
    }

    # Without additional reasoning
    tech = ChainOfThought(schema, enable_reasoning=False)
    prompt = tech.make_prompt("test text")
    # CoT already has reasoning built-in, but shouldn't have the extra instruction
    assert "explanation for your classification" not in prompt.lower()

    # With additional reasoning instruction
    tech = ChainOfThought(schema, enable_reasoning=True, max_reasoning_chars=150)
    prompt = tech.make_prompt("test text")
    assert "explanation for your classification" in prompt.lower()
    assert "150 characters" in prompt


def test_technique_output_keys_zeroshot():
    """Test dynamic output_keys based on configuration for ZeroShot."""
    schema = {"classes": [{"name": "A", "description": "Class A"}]}

    # Without reasoning
    tech = ZeroShot(schema, enable_reasoning=False)
    assert "confidence" in tech.output_keys
    assert "reasoning" not in tech.output_keys

    # With reasoning
    tech = ZeroShot(schema, enable_reasoning=True)
    assert "confidence" in tech.output_keys
    assert "reasoning" in tech.output_keys


def test_technique_output_keys_fewshot():
    """Test dynamic output_keys based on configuration for FewShot."""
    schema = {
        "classes": [{"name": "A", "description": "Class A"}],
        "examples": [{"query": "example", "classification": "A"}]
    }

    # Without reasoning
    tech = FewShot(schema, enable_reasoning=False)
    assert "confidence" in tech.output_keys
    assert "reasoning" not in tech.output_keys

    # With reasoning
    tech = FewShot(schema, enable_reasoning=True)
    assert "confidence" in tech.output_keys
    assert "reasoning" in tech.output_keys


def test_technique_output_keys_cot():
    """Test dynamic output_keys based on configuration for CoT."""
    schema = {
        "classes": [{"name": "A", "description": "Class A"}],
        "examples": [{"query": "example", "classification": "A", "reason": "Because"}]
    }

    # Without additional reasoning - CoT has 'reason' by default
    tech = ChainOfThought(schema, enable_reasoning=False)
    assert "confidence" in tech.output_keys
    assert "reason" in tech.output_keys
    assert "reasoning" not in tech.output_keys

    # With additional reasoning
    tech = ChainOfThought(schema, enable_reasoning=True)
    assert "confidence" in tech.output_keys
    assert "reason" in tech.output_keys
    assert "reasoning" in tech.output_keys


def test_reasoning_effort_none_removal():
    """Test that reasoning_effort=None is removed from kwargs."""
    from atap_llm_classifier.models import LiteLLMCompletionArgs, LiteLLMMessage, LiteLLMRole

    args = LiteLLMCompletionArgs(
        model="gpt-4o-mini",
        messages=[LiteLLMMessage(content="test", role=LiteLLMRole.USER)],
        temperature=1.0,
        top_p=1.0,
        n=1,
        reasoning_effort=None,
    )
    kwargs = args.to_kwargs()
    assert "reasoning_effort" not in kwargs, "reasoning_effort=None should be removed"


def test_reasoning_effort_preserved():
    """Test that reasoning_effort values are preserved in kwargs."""
    from atap_llm_classifier.models import LiteLLMCompletionArgs, LiteLLMMessage, LiteLLMRole

    for effort in ["low", "medium", "high"]:
        args = LiteLLMCompletionArgs(
            model="gpt-4o-mini",
            messages=[LiteLLMMessage(content="test", role=LiteLLMRole.USER)],
            temperature=1.0,
            top_p=1.0,
            n=1,
            reasoning_effort=effort,
        )
        kwargs = args.to_kwargs()
        assert kwargs["reasoning_effort"] == effort
