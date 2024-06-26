def get_settings():
    from atap_llm_classifier.settings import get_env_settings

    return get_env_settings()


def get_view_settings():
    from atap_llm_classifier.views.settings import get_settings

    return get_settings()
