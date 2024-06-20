# https://github.com/ollama/ollama/blob/main/docs/api.md
import httpx

LIST_LOCAL_MODEL_PATH: str = "/api/tags"


def get_available_models(endpoint: str) -> list[str]:
    res: httpx.Response = httpx.get(endpoint + LIST_LOCAL_MODEL_PATH)
    res.raise_for_status()
    return [m["name"] for m in res.json()["models"]]
