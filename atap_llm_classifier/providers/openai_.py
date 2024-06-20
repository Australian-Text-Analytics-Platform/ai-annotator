from openai import OpenAI
from openai.pagination import SyncPage


def get_available_models_for_user(
    api_key: str,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Retrieves models available to the user.

    Returns set of base models, set of (finetune, finetune base) models.
    """
    client = OpenAI(api_key=api_key)
    paginator: SyncPage = client.models.list()
    all_models: set[str] = {
        model.id for page in paginator.iter_pages() for model in page.data
    }

    finetune_models: set[str] = {
        model for model in all_models if model.startswith("ft:")
    }
    base_models: set[str] = all_models.difference(finetune_models)
    finetune_ftbase_models: set[tuple[str, str]] = {
        (ft_model, ft_model.split(":")[1]) for ft_model in finetune_models
    }
    return base_models, finetune_ftbase_models
