from atap_llm_classifier.providers import LLMProvider


def test_all_providers_have_available_models():
    for provider in LLMProvider:
        if provider is LLMProvider.OPENAI_AZURE_SIH:
            # skipped - no models specified at the moment
            continue
        props = provider.properties
        assert len(props.models) > 0, f"No available models for provider: {provider}."
