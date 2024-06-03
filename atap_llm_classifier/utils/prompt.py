"""prompt_utils.py"""

import tiktoken


def validate_context_window(prompt_str: str, model: str) -> bool:
    return False


def count_subwords(prompt: str, model: str) -> int:
    """Return the number of subwords in the prompt given model.

    Args:
        prompt:
        model:

    Returns:

    Raises:
        KeyError if encoding not found.
    """
    encoding = tiktoken.encoding_for_model(model)
    encoded = encoding.encode(prompt)
    return len(encoded)
