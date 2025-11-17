from functools import lru_cache
import math

import pandas as pd
from litellm import model_cost

__all__ = [
    "get_available_models",
    "prettify_model_list",
]


@lru_cache()
def load_model_cost_as_df() -> pd.DataFrame:
    return pd.DataFrame.from_dict(model_cost, orient="index")


def prettify_model_list(html: bool = False) -> str:
    df = load_model_cost_as_df()
    df.sort_values(by="litellm_provider", inplace=True)

    df = df[df["mode"].isin(["chat", "completion", "embedding"])]
    df = df.loc[
        :,
        [
            "litellm_provider",
            "mode",
            "max_tokens",
            "max_input_tokens",
            "max_output_tokens",
            "input_cost_per_token",
            "output_cost_per_token",
        ],
    ]

    # Set the width of each column to prevent truncation
    pd.set_option("display.max_colwidth", 150)
    # Optionally, set the width of the display in characters
    pd.set_option("display.width", 1000)

    pd.set_option("display.max_rows", len(df))

    if html:
        return str(
            df.reset_index().rename(columns={"index": "model"}).to_html(index=False)
        )
    else:
        return str(df)


def get_available_models(provider: str) -> list[str]:
    df = load_model_cost_as_df()
    mask = df["litellm_provider"] == provider
    mask = df["mode"].isin(["chat", "completion"]) & mask
    return df[mask].index.tolist()


def get_context_window(model: str) -> int | None:
    df = load_model_cost_as_df()
    max_input_tokens = df.loc[model, "max_input_tokens"]
    # Handle NaN or missing values
    if pd.isna(max_input_tokens) or (isinstance(max_input_tokens, float) and math.isnan(max_input_tokens)):
        return None
    return int(max_input_tokens)


def get_price(model: str) -> tuple[float, float]:
    df = load_model_cost_as_df()
    return (
        float(df.loc[model, "input_cost_per_token"]),
        float(df.loc[model, "output_cost_per_token"]),
    )


def add_ollama_provider_prefix(model: str) -> str:
    return "ollama_chat/" + model
