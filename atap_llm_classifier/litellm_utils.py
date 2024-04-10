import pandas as pd
from litellm import model_cost


def pretty_print_model_list(html: bool = False):
    df = pd.DataFrame.from_dict(model_cost, orient="index")
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
        print(df.reset_index().rename(columns={"index": "model"}).to_html(index=False))
    else:
        print(df)
