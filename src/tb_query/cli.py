import json

import click

from tb_query.core import (
    ValidationError,
    calculate_correlation,
    find_event_files,
    get_all_tags,
    get_tag_statistics,
    get_tag_steps,
    query_tensorboard,
)


@click.group()
def cli():
    """A CLI tool for querying TensorBoard event files."""
    pass


@cli.command()
@click.argument("event_file", type=click.Path(exists=True))
@click.option(
    "--tags",
    multiple=True,
    help="A list of scalar tags to query. If not provided, all available scalar tags are queried.",
)
@click.option("--start_step", type=int, default=None, help="The starting step (inclusive) to filter data from.")
@click.option("--end_step", type=int, default=None, help="The ending step (inclusive) to filter data to.")
def query(event_file: str, tags: list[str], start_step: int | None, end_step: int | None) -> None:
    """
    Query scalar data from a TensorBoard event file and output as JSON.

    Example usage:

        tb_query query path/to/events.out.tfevents.12345
        tb_query query path/to/events.out.tfevents.12345 --tags loss --tags accuracy
        tb_query query path/to/events.out.tfevents.12345 --start_step 100 --end_step 200
    """
    try:
        click.echo(query_tensorboard(event_file, list(tags) if tags else None, start_step, end_step))
    except ValidationError as e:
        click.echo(json.dumps(e))


@cli.command()
@click.argument("event_file", type=click.Path(exists=True))
@click.option(
    "--filter",
    "filters",
    multiple=True,
    help="Filter tags by these strings. Tags containing any of these strings will be included.",
)
def tags(event_file: str, filters: list[str]) -> None:
    """
    Get all available scalar tags from a TensorBoard event file with optional filtering.

    Example usage:

        tb_query tags path/to/events.out.tfevents.12345
        tb_query tags path/to/events.out.tfevents.12345 --filter loss
        tb_query tags path/to/events.out.tfevents.12345 --filter loss --filter accuracy
    """
    try:
        click.echo(json.dumps(get_all_tags(event_file, list(filters) if filters else None)))
    except ValidationError as e:
        click.echo(json.dumps({"error": e.message}), err=True)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def find(directory: str) -> None:
    """
    Find all TensorBoard event files in the specified directory and its subdirectories.

    Example usage:
        tb_query find path/to/logs
    """
    try:
        click.echo(json.dumps(find_event_files(directory)))
    except ValidationError as e:
        click.echo(json.dumps({"error": e.message}), err=True)


@cli.command()
@click.argument("event_file", type=click.Path(exists=True))
@click.option("--tags", multiple=True, required=True, help="A list of scalar tags to get steps for.")
def steps(event_file: str, tags: list[str]) -> None:
    """
    Get the steps for each specified tag from a TensorBoard event file.

    Example usage:
        tb_query steps path/to/events.out.tfevents.12345 --tags loss accuracy
    """
    try:
        click.echo(get_tag_steps(event_file, list(tags)))
    except ValidationError as e:
        click.echo(json.dumps({"error": e.message}), err=True)


@cli.command()
@click.argument("event_file", type=click.Path(exists=True))
@click.option("--tags", multiple=True, required=True, help="A list of scalar tags to get statistics for.")
def stats(event_file: str, tags: list[str]) -> None:
    """
    Get statistical measures (min, max, mean, std) for each specified tag's values.

    Example usage:
        tb_query stats path/to/events.out.tfevents.12345 --tags loss --tags accuracy
    """
    try:
        click.echo(get_tag_statistics(event_file, list(tags)))
    except ValidationError as e:
        click.echo(json.dumps({"error": e.message}), err=True)


@cli.command()
@click.argument("event_file", type=click.Path(exists=True))
@click.argument("tags", type=str)
@click.option("--start_step", type=int, default=None, help="The starting step (inclusive) to filter data from.")
@click.option("--end_step", type=int, default=None, help="The ending step (inclusive) to filter data to.")
@click.option("--display-interpretation", type=bool, default=False, help="Display user friendly interpretation.")
@click.option("--rounding", type=int, default=4, help="Rounding decimal places for correlation number")
def correlation(
    event_file: str,
    tags: str,
    start_step: int | None,
    end_step: int | None,
    display_interpretation: bool = False,
    rounding: int = 4,
) -> None:
    """
    Calculate the correlation between scalar tags in a TensorBoard event file and output as JSON.

    Example usage:

        tb_query correlation path/to/events.out.tfevents.12345
        tb_query correlation path/to/events.out.tfevents.12345 --tags loss --tags accuracy
        tb_query correlation path/to/events.out.tfevents.12345 --start_step 100 --end_step 200
    """
    tag_set = {i.strip() for i in tags.split(",")}
    try:
        result_query = query_tensorboard(event_file, None, start_step, end_step)
        click.echo(calculate_correlation(result_query, tag_set, rounding, display_interpretation))
    except ValidationError as e:
        click.echo(json.dumps({"error": e.message}), err=True)


if __name__ == "__main__":
    cli()
