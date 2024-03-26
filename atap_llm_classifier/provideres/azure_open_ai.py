def list_models() -> list[str]:
    # todo: this should really be based on your API key.
    return [
        "azure/gpt-3.5-turbo",
        "azure/gpt-4",
    ]


def validate_api_key(key: str) -> bool:
    pass
