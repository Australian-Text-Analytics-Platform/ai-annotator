import functools
import typer

from atap_llm_classifier.litellm_utils import pretty_print_model_list

# subcommand: litellm

cli_litellm: typer.Typer = typer.Typer()


@functools.wraps(pretty_print_model_list)
@cli_litellm.command(name="list-models")
def list_models(*args, **kwargs):
    return pretty_print_model_list(*args, **kwargs)


# main cli
cli: typer.Typer = typer.Typer()
cli.add_typer(cli_litellm, name="litellm")


def main():
    cli()
