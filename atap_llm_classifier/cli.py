from enum import Enum
from typing import Callable

try:
    import fire
except ImportError as e:
    raise ImportError(
        "Please install with 'cli' extras. e.g. pip install 'atap-llm-classifier[cli]'"
    )


class Command(Enum):
    LITELLM_LIST_MODELS: str = "litellm.list_models"


def main():
    from atap_llm_classifier.litellm_utils import pretty_print_model_list

    fire.Fire(
        {
            "litellm": {
                "list_models": pretty_print_model_list,
            },
        }
    )


def _get_fn(cmd: Command) -> Callable:
    match cmd:
        case Command.LITELLM_LIST_MODELS:
            from atap_llm_classifier.litellm_utils import pretty_print_model_list

            return pretty_print_model_list
        case _:
            raise NotImplementedError()


if __name__ == "__main__":
    main()
