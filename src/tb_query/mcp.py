from contextlib import contextmanager
from typing import Annotated, Mapping, NoReturn, cast

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, RootModel

from tb_query.core import (
    TB_QUERY_EVENTS_PATH,
    ValidationError,
    calculate_correlation,
    find_event_files,
    get_all_tags,
    get_tag_statistics,
    get_tag_steps,
    query_tensorboard,
)


mcp = FastMCP(name="TB-Query MCP Server")


@contextmanager
def catch_validation_error():
    try:
        yield
    except ValidationError as e:
        raise ToolError(e.message)


class StepValue(BaseModel):
    step: int
    value: float


class QueryResponseItem(BaseModel):
    tag: str = Field(description="Name of the tag.")
    steps: list[StepValue]


class TagStepItem(BaseModel):
    tag: str = Field(description="Name of the tag.")
    steps: list[int]


class TagStepResponse(BaseModel):
    results: list[TagStepItem]


class TagStats(BaseModel):
    min: float = Field(description="Minimum value of the tag measurements.")
    max: float = Field(description="Maximum value of the tag measurements.")
    mean: float = Field(description="Mean (average) value of the tag measurements.")
    std: float = Field(description="Standard deviation of the tag measurements.")
    count: int = Field(description="Number of measurements for the tag.")


class TagStatsItem(BaseModel):
    tag: str = Field(description="Name of the tag.")
    stats: TagStats = Field(description="Statistics of the tags.")


class TagStatsResponse(BaseModel):
    results: list[TagStatsItem]


class EventFileItem(BaseModel):
    path: str
    created_at: str


class FindEventFileResponse(BaseModel):
    results: list[EventFileItem] = Field(description="List event found event files.")


class QueryResponse(BaseModel):
    results: list[QueryResponseItem]


class TagsResponse(BaseModel):
    results: list[str] = Field(description="List of tags available in the Tensorboard event file.")


def _validate_parameter(value: str | None, parameter_name: str) -> str | NoReturn:
    value = value.strip() if value else None
    if not value:
        raise ToolError({"error": f"Parameter `{parameter_name}` is required and cannot be blank"})

    return value


@mcp.tool()
def query(
    event_file: Annotated[str, Field(description="Path of the tensorboard event file.")],
    tags: Annotated[
        list[str],
        Field(description="List of tags to show. If not provided, all the tags will be queried.", default_factory=list),
    ],
    start_step: Annotated[int | None, Field(description="Query the scalar data starting with this step.")] = None,
    end_step: Annotated[int | None, Field(description="Query the scalar data until this step.")] = None,
) -> QueryResponse:
    """
    Query scalar data from a TensorBoard event file.

    This will return all the data inside the Tensorboard event files. The result for a long training is very big and can
    consume all your context limit if not properly filtered by `tags` and `start_step` and end_step.

    - To know the available event files you can use `find_events` tool to get all the available event files.
    - To know the available tags of a Tensorboard file, you may use `list_tags` which gives you all the available
        scalar tags in the event file.
    - To know all the available training steps for each tag, you can use `tag_steps` tool.
    - Use start_step and end_step to ask for a range of data.
    - Try to use tags explicitly, otherwise the output can be huge.
    """
    event_file = _validate_parameter(event_file, "event_file")
    if not tags:
        tags = []

    with catch_validation_error():
        results = query_tensorboard(event_file=event_file, tags=tags, start_step=start_step, end_step=end_step)

    return QueryResponse(
        results=[
            QueryResponseItem(
                tag=k,
                steps=[
                    StepValue(
                        step=step["step"],
                        value=step["value"],
                    )
                    for step in v
                ],
            )
            for k, v in results.items()
        ]
    )


@mcp.tool()
def list_tags(
    event_file: Annotated[str, Field(description="Path of the tensorboard event file.")],
    filters: Annotated[
        list[str],
        Field(
            description=(
                "List of filters to match the tag names. If a filter is given, any tag name that includes the given "
                "characters in the filter will be returned."
            )
        ),
    ] = None,  # type: ignore
) -> TagsResponse:
    """
    Get all available scalar tags from a TensorBoard event file with optional filtering.

    - To search/filter the returning tags, you can pass a list of string to the `filters` parameter, which will
        only return the tags that contain those filter strings.
    """
    event_file = _validate_parameter(event_file, "event_file")

    with catch_validation_error():
        results = get_all_tags(event_file=event_file, filters=filters)

    return TagsResponse(results=results["tags"])


@mcp.tool()
def find_events(
    directory: Annotated[
        str,
        Field(description="Path of the directory to find all the tensorboard events file in."),
    ],
) -> FindEventFileResponse:
    """
    Find all TensorBoard event files in the specified directory and its subdirectories.

    Use this endpoint when you don't know the available events or user has not given you any event file name to analyze.
    """
    directory = _validate_parameter(directory, "directory")

    with catch_validation_error():
        results = find_event_files(directory=directory)

    return FindEventFileResponse(
        results=[EventFileItem(path=i["path"], created_at=i["created_at"]) for i in results["event_files"]]
    )


@mcp.resource(
    uri="resource://event-files",
    name="EventFiles",
    description="Shows all the event files available since last training run.",
    mime_type="application/json",
)
def event_files() -> FindEventFileResponse:
    """ "
    Shows all the event files available for analyzing since the last run. Usually, the newest one is
    what you should be analyzing.
    """
    directory = TB_QUERY_EVENTS_PATH

    if not directory:
        raise ToolError(
            {
                "error": (
                    "No Tensorboard Events directory is set to look for event files. "
                    "Hint: Set `TB_QUERY_EVENTS_PATH` environment variable."
                )
            }
        )

    results = find_event_files(directory=directory)["event_files"]

    return FindEventFileResponse(results=[EventFileItem(path=i["path"], created_at=i["created_at"]) for i in results])


@mcp.tool()
def tag_steps(
    event_file: Annotated[str, Field(description="Path of the tensorboard event file.")],
    tags: Annotated[list[str], Field(description="List of tags to show the steps for.")],
) -> TagStepResponse:
    """
    Get the steps for each specified tag from a TensorBoard event file.

    - Pass tags to get return steps for only specified tags, otherwise the output can be quite large.
    """
    event_file = _validate_parameter(event_file, "event_file")
    with catch_validation_error():
        results = get_tag_steps(event_file=event_file, tags=tags)

    return TagStepResponse(results=[TagStepItem(tag=k, steps=v) for k, v in results.items()])


@mcp.tool(description="Get statistical measures (min, max, mean, std) for each specified tag's values.")
def tag_stats(
    event_file: Annotated[str, Field(description="Path of the tensorboard event file.")],
    tags: Annotated[list[str], Field(description="List of tags to show the stats for.")],
) -> TagStatsResponse:
    """
    Get statistical measures (min, max, mean, std) for each specified tag's values.

    - Optionally, pass tags to get stats for only specified tags.
    """
    event_file = _validate_parameter(event_file, "event_file")

    with catch_validation_error():
        results = get_tag_statistics(event_file=event_file, tags=tags)

    return TagStatsResponse(
        results=[
            TagStatsItem(
                tag=k,
                stats=TagStats(
                    min=cast(float, v["min"]),
                    max=cast(float, v["max"]),
                    mean=cast(float, v["mean"]),
                    std=cast(float, v["std"]),
                    count=cast(int, v["count"]),
                ),
            )
            for k, v in results.items()
        ]
    )


class CorrelationResponse(RootModel[Mapping[str, Mapping[str, Mapping[str, str | float] | float]]]):
    root: Mapping[str, Mapping[str, Mapping[str, str | float] | float]] = Field(
        description=(
            "Dynamic metric data where top-level keys are metric group names "
            "(e.g., 'rewards/scaled_reward_unrealized_profit'), and values are "
            "dictionaries of sub-metrics (e.g., 'actions/Buy', 'info/position') "
            "mapped to float values representing normalized scores, losses, or correlations."
        )
    )


@mcp.tool()
def correlation(
    event_file: Annotated[str, Field(description="Path of the tensorboard event file.")],
    tags: Annotated[list[str], Field(description="List of tags to show correlation for with other tags.")],
    start_step: Annotated[int | None, Field(description="Query the scalar data starting with this step.")] = None,
    end_step: Annotated[int | None, Field(description="Query the scalar data until this step.")] = None,
) -> CorrelationResponse:
    """
    Calculate the correlation between scalar tags in a TensorBoard event file

    It will provide correlation for the given tag(s) with other tags in the tensorboard data.
    """
    if not tags:
        raise ToolError("Parameter `tags` is required.")

    with catch_validation_error():
        result_query = query_tensorboard(event_file, None, start_step, end_step)
        result = calculate_correlation(result_query, set(tags))

    return CorrelationResponse(result)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
