"""prompt_utils.py"""

import tiktoken


def count_tokens_for_openai(prompt: str, model: str) -> int:
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
