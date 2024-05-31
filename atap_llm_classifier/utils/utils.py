import traceback

__all__ = [
    "format_exception",
    "make_dummy_request",
    "is_jupyter_context",
]

from functools import lru_cache

import litellm
from litellm import ModelResponse

from atap_llm_classifier.models import LiteLLMMessage, LiteLLMRole, LiteLLMArgs


def format_exception(e: Exception) -> str:
    format_str: str = "{}:{}\t{}"
    tb = list(iter(traceback.extract_tb(e.__traceback__)))
    if len(tb) > 0:
        frame = tb[-1]
        return format_str.format(frame.filename, frame.lineno, e)
    else:
        return str(e)


def is_jupyter_context() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if "ZMQInteractiveShell" in shell:
            return True
        else:
            return False
    except NameError:
        return False


@lru_cache
def make_dummy_request(
    model: str,
    api_key: str,
) -> ModelResponse:
    msg = LiteLLMMessage(content="Say Yes.", role=LiteLLMRole.USER)
    return litellm.completion(
        **LiteLLMArgs(
            model=model,
            messages=[msg],
            temperature=0,
            top_p=1.0,
            n=1,
            api_key=api_key,
        ).to_kwargs(),
        max_tokens=10,
    )
