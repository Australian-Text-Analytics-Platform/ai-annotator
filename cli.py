import functools
import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Annotated, Optional

import pandas as pd
import typer

from atap_corpus import Corpus
from loguru import logger

from atap_llm_classifier import pipeline, config
from atap_llm_classifier.models import LLMConfig
from atap_llm_classifier.modifiers import Modifier
from atap_llm_classifier.providers import LLMProvider
from atap_llm_classifier.techniques import Technique
from atap_llm_classifier.utils.litellm_ import prettify_model_list

# cmd: classify
cmd_classify: typer.Typer = typer.Typer()


@cmd_classify.command(name="one")
def classify_one(
    text: str,
    out_dir: str,
    provider: Annotated[LLMProvider, typer.Option(case_sensitive=False)],
    model: str,
    user_schema: str,
    technique: Annotated[
        Technique, typer.Option(case_sensitive=False)
    ] = Technique.ZERO_SHOT,
    modifier: Annotated[
        Modifier, typer.Option(case_sensitive=False)
    ] = Modifier.NO_MODIFIER,
    temperature: Annotated[Optional[float], typer.Argument()] = None,
    top_p: Annotated[Optional[float], typer.Argument()] = None,
    api_key: Annotated[Optional[str], typer.Argument()] = None,
    endpoint: Annotated[Optional[str], typer.Argument()] = None,
):
    raise NotImplementedError()


@cmd_classify.command(name="batch")
def classify_batch(
    dataset: Annotated[str, typer.Option()],
    column: Annotated[str, typer.Option()],
    out_dir: Annotated[str, typer.Option()],
    provider: Annotated[LLMProvider, typer.Option(case_sensitive=False)],
    model: Annotated[str, typer.Option()],
    user_schema: Annotated[str, typer.Option()],
    technique: Annotated[
        Technique, typer.Option(case_sensitive=False)
    ] = Technique.ZERO_SHOT,
    modifier: Annotated[
        Modifier, typer.Option(case_sensitive=False)
    ] = Modifier.NO_MODIFIER,
    temperature: Annotated[Optional[float], typer.Option()] = None,
    top_p: Annotated[Optional[float], typer.Option()] = None,
    api_key: Annotated[Optional[str], typer.Option()] = None,
    endpoint: Annotated[Optional[str], typer.Option()] = None,
):
    # config.mock.enabled = True
    dataset: Path = Path(dataset)
    if not dataset.exists():
        print_err("dataset does not exist.")
        exit(1)
    if not dataset.is_file():
        print_err("dataset is not a file.")
        exit(1)

    out_dir: Path = Path(out_dir)
    if out_dir.is_file():
        print_err("out_dir is a file.")
        exit(1)
    try:
        if Path(user_schema).exists():
            with open(user_schema, "r") as h:
                user_schema = json.load(h)
        else:
            user_schema = json.loads(user_schema)
    except Exception as e:
        print_err("User schema must be json or path to json.", e)
        exit(1)

    match dataset.suffix:
        case ".csv":
            df = pd.read_csv(dataset.absolute())
        case ".xlsx":
            df = pd.read_excel(dataset.absolute())
        case _:
            raise NotImplementedError(f"{dataset.suffix} is not supported.")

    if not technique.prompt_maker_cls.is_validate_user_schema(user_schema):
        print_err(f"Invalid user schema for given technique={technique}.")
        exit(1)

    corpus: Corpus = Corpus.from_dataframe(df=df, col_doc=column)

    default: LLMConfig = LLMConfig()
    default_temp = default.temperature
    default_top_p = default.top_p
    llm_config: LLMConfig = LLMConfig(
        temperature=temperature if temperature is not None else default_temp,
        top_p=top_p if top_p is not None else default_top_p,
    )

    provider_props = provider.properties
    if endpoint is not None:
        provider_props = provider_props.with_endpoint(endpoint)
    if api_key is not None:
        provider_props = provider_props.with_api_key(api_key)

    try:
        model_props = provider_props.get_model_props(model)
    except ValueError as e:
        print_err("Invalid model.", e)
        exit(1)

    print(f"dataset: {dataset.absolute()}")
    print(f"column: {column}")
    print(f"out_dir: {out_dir.absolute()}")
    print("-- model --")
    pprint(model_props.dict())
    print("-- llm configurations --")
    pprint(llm_config.dict())
    print("-- user schema --")
    pprint(user_schema)

    if not typer.confirm("confirm? (y/n): "):
        raise typer.Abort()

    completed: int = 0
    total: int = len(corpus)

    def on_results(res):
        nonlocal completed
        completed += 1
        logger.info(f"PROGRESS UPDATE: classified {completed}/{total} documents.")

    batch_results = pipeline.batch(
        corpus=corpus,
        model_props=model_props,
        llm_config=llm_config,
        technique=technique,
        user_schema=user_schema,
        modifier=modifier,
        on_result_callback=on_results,
    )
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results.json", "w") as h:
        h.write(batch_results.model_dump_json(exclude=["user_schema"]))
    # note - for some reason it aint serialising user_schema.
    with open(out_dir / "user_schema.json", "w") as h:
        h.write(batch_results.user_schema.model_dump_json())
    _ = corpus.serialise(out_dir / "corpus.zip")
    corpus.to_dataframe().to_csv(out_dir / "corpus.csv")
    print("Saved to ", out_dir.absolute())
    return


# cmd: info
cmd_info: typer.Typer = typer.Typer()


@cmd_info.command(name="info")
def info():
    pass


# cmd: litellm


cmd_litellm: typer.Typer = typer.Typer()


@functools.wraps(prettify_model_list)
@cmd_litellm.command(name="list-models")
def list_models(*args, **kwargs):
    print(prettify_model_list(*args, **kwargs))


# helpers
def print_err(msg: str, e: Exception | None = None):
    if e is None:
        print(f"{msg}", file=sys.stderr)
    else:
        print(f"{msg}\tErr: {e}", file=sys.stderr)


# main cli
cli: typer.Typer = typer.Typer()
cli.add_typer(cmd_classify, name="classify")
cli.add_typer(cmd_litellm, name="litellm")


def main():
    cli()
