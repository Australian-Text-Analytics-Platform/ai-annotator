from enum import Enum

from openai import AsyncClient
from openai.types import Model
from litellm.utils import check_valid_key


# aclient = AsyncClient(
#     api_key=api_settings.OPENAI_API_KEY,
# )


class OpenAIOwnedBy(Enum):
    OPENAI: str = "openai"

    def __str__(self):
        return self.value


async def list_gpt_models(client: AsyncClient, openai_only: bool = True) -> list[Model]:
    """

    Args:
        client:
        openai_only:

    Returns:
        list of GPT only openai's Model objects.
    """
    models: list[Model] = list()
    async for model in client.models.list():
        if openai_only and model.owned_by != OpenAIOwnedBy.OPENAI.value:
            continue

        # allow fine-tuned model to be shown.
        if model.id.startswith("gpt-") or model.id.startswith("ft:gpt-"):
            models.append(model)
    return models


def list_models() -> list[str]:
    # todo: this should really be based on your provided API key.
    return [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-32k",
    ]


def validate_api_key(key: str) -> bool:
    return check_valid_key(model="gpt-3.5-turbo", api_key=key)
