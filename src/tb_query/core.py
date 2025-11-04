"""
Utilities for reading, querying, and analyzing TensorBoard event files.

This module provides a set of functions to interact with TensorBoard's
`events.out.tfevents.*` files directly, without needing a running TensorBoard
server. It allows for:
- Querying scalar data with step and tag filtering.
- Finding all event files in a directory.
- Retrieving all available scalar tags.
- Calculating statistics (min, max, mean, std) for specific tags.
- Calculating correlations between different scalar tags.

It relies on the `tensorboard.backend.event_processing.event_accumulator`
module for the core event file parsing.
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Mapping, Optional, TypedDict, cast

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Suppress excessive TensorFlow logging (INFO and WARNING messages)
# '2' = all INFO and WARNING messages are not printed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Environment variable (optional) to specify a default events path.
# Not used directly in this module's functions but available for consumers.
TB_QUERY_EVENTS_PATH = os.environ.get("TB_QUERY_EVENTS_PATH", None)


class ScalarTag(TypedDict):
    """
    A TypedDict representing a single scalar data point from TensorBoard.

    Attributes:
        step (int): The training or evaluation step number.
        value (int | float): The logged scalar value at that step.
    """

    step: int
    value: int | float


class ValidationError(Exception):
    """
    Custom exception raised for file validation or loading errors.

    Used when an event file cannot be found or fails to be parsed by
    the EventAccumulator.
    """

    def __init__(self, message: str) -> None:
        """
        Initializes the ValidationError.

        :param message: A human-readable error message.
        """
        self.message = message


def _get_event_tags(event_file: str) -> EventAccumulator:
    """
    Internal helper to load and reload a TensorBoard event file.

    Validates the file path and attempts to load it using EventAccumulator.
    This function is configured to load *all* scalar data by setting
    `size_guidance` to 0.

    :param event_file: The path to the `.tfevents` file.
    :raises ValidationError: If the file is not found or fails to load (e.g., if the file is corrupt).
    :returns A loaded and reloaded EventAccumulator object containing the data from the event file.
    """
    if not os.path.exists(event_file):
        raise ValidationError(f"File not found: {event_file}")

    try:
        # Initialize the EventAccumulator.
        # size_guidance={event_accumulator.SCALARS: 0} tells it to load
        # all scalar events, overriding any default downsampling.
        acc = event_accumulator.EventAccumulator(
            event_file, size_guidance={event_accumulator.SCALARS: 0}  # Load all scalar data
        )
        # Reload() parses the file and populates the accumulator with data.
        acc.Reload()
    except Exception as e:
        # Catch generic exceptions during loading (e.g., corrupt file, permissions)
        raise ValidationError(f"Failed to load event file: {str(e)}")

    return acc


def query_tensorboard(
    event_file: str, tags: Optional[list[str]] = None, start_step: int | None = None, end_step: int | None = None
) -> dict[str, list[ScalarTag]]:
    """
    Extracts scalar data from a TensorBoard event file for specific tags and step range.

    :param event_file: Path to the TensorBoard event file.
    :param tags: A list of scalar tags to extract. If None or empty, all available scalar tags are queried.
        Defaults to None.
    :param start_step: The minimum step (inclusive) to include.
        If None, filtering starts from the first step. Defaults to None.
    :param end_step: The maximum step (inclusive) to include.
        If None, filtering includes steps up to the last step. Defaults to None.

    :returns: A dictionary where keys are the queried tag names and values are
        lists of ScalarTag objects (containing 'step' and 'value') within the specified step range.

    :raises ValidationError: If the event_file cannot be loaded (propagated from _get_event_tags).
    """
    # Load the event file using the helper
    acc = _get_event_tags(event_file)
    # Get a list of all available scalar tags in the file
    available_tags = acc.Tags()["scalars"]

    # If no specific tags are requested, default to using all available scalar tags
    if not tags:
        tags_to_query = available_tags
    else:
        tags_to_query = tags

    # Use defaultdict to simplify appending to lists for each tag
    output_data: dict[str, list[ScalarTag]] = defaultdict(list)

    for tag in tags_to_query:
        if tag not in available_tags:
            # You might want to log a warning here, but for now, we silently ignore
            # requested tags that do not exist in the file.
            continue

        # Retrieve all scalar events for the given tag
        events = acc.Scalars(tag)

        for event in events:
            step = event.step
            # Apply step filtering
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue

            # If the event is within the step range, add it to the output
            output_data[tag].append({"step": step, "value": event.value})

    return output_data


def get_all_tags(event_file: str, filters: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Gets all available scalar tags from a TensorBoard event file with optional filtering.

    :param event_file: Path to the TensorBoard event file.
    :param filters: A list of filter strings.
        If provided, only tags containing at least one of these strings will be returned.
        Defaults to None (no filtering).

    :returns: A dictionary with a single key "tags", containing a list of the filtered tag names.
    :raises ValidationError: If the event_file cannot be loaded.
    """
    acc = _get_event_tags(event_file)
    available_tags = acc.Tags()["scalars"]

    if filters:
        filtered_tags = [tag for tag in available_tags if any(filter_str in tag for filter_str in filters)]
    else:
        filtered_tags = available_tags

    return {"tags": filtered_tags}


def find_event_files(directory: str) -> dict[str, list[dict[str, str]]]:
    """
    Finds all TensorBoard event files in the specified directory and its subdirectories.

    Searches recursively for files starting with "events.out.tfevents".

    :param directory: Path to the root directory to search.

    :returns: A dictionary with a single key "event_files".
        The value is a list of dictionaries, where each dictionary
        represents one file and contains:
            - "path": The relative path to the file from the current working directory.
            - "created_at": An ISO 8601 formatted timestamp of the file's creation (or modification time as a fallback).
        The list is sorted by creation time (newest first).

    :raises ValidationError: If the specified directory does not exist.
    """
    if not os.path.exists(directory):
        raise ValidationError(f"Directory not found: {directory}")

    event_files = []
    # Recursively walk the directory tree
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the TensorBoard event file pattern
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)

                # Get creation time (os.path.getctime)
                # This may return modification time on some Unix systems.
                try:
                    created_at = os.path.getctime(file_path)
                except (OSError, ValueError):
                    # Fallback to modification time if creation time is unavailable
                    created_at = os.path.getmtime(file_path)

                created_at_iso = datetime.fromtimestamp(created_at).isoformat()

                event_files.append({"path": file_path, "created_at": created_at_iso})

    # Sort the list of files by creation time, descending (newest first)
    event_files.sort(key=lambda x: x["created_at"], reverse=True)

    return {"event_files": event_files}


def get_tag_steps(event_file: str, tags: list[str]) -> dict[str, list[int]]:
    """
    Gets *only* the steps for each specified tag from a TensorBoard event file.

    This is a lighter-weight query than `query_tensorboard` if only step
    information is needed.

    :param event_file: Path to the TensorBoard event file.
    :param tags: A list of tags to extract steps for.

    :returns: A dictionary where keys are tag names and values are lists of step integers.
        If a tag is not found, its value will be an empty list.

    :raises ValidationError: If the event_file cannot be loaded.
    """
    acc = _get_event_tags(event_file)
    available_tags = acc.Tags()["scalars"]
    output_data: dict[str, list[int]] = {}

    for tag in tags:
        if tag not in available_tags:
            # If a requested tag doesn't exist, return it with an empty list
            output_data[tag] = []
            continue

        # Get scalar events and extract just the step component
        events = acc.Scalars(tag)
        steps = [event.step for event in events]
        output_data[tag] = steps

    return output_data


def get_tag_statistics(event_file: str, tags: list[str]) -> dict[str, dict[str, float | int | str]]:
    """
    Gets statistical measures (min, max, mean, std) for each specified tag's values.

    :param event_file: Path to the TensorBoard event file.
    :param tags: A list of tags to extract statistics for.

    :raises: A dictionary where keys are tag names.
        Each value is another dictionary containing the keys:
        "min", "max", "mean", "std", and "count".
        If a tag is not found, the value will be {"error": "Tag not found"}.
        If a tag has no values, the value will be {"error": "No values found for tag"}.

    :raises ValidationError: If the event_file cannot be loaded.
    """
    acc = _get_event_tags(event_file)
    available_tags = acc.Tags()["scalars"]
    output_data: dict[str, dict[str, float | int | str]] = {}

    for tag in tags:
        if tag not in available_tags:
            output_data[tag] = {"error": "Tag not found"}
            continue

        # Get scalar events and extract just the value component
        events = acc.Scalars(tag)
        values = [event.value for event in events]

        if not values:
            # Handle tags that exist but have no data points
            output_data[tag] = {"error": "No values found for tag"}
            continue

        # Calculate statistics using numpy for efficiency
        output_data[tag] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": len(values),
        }

    return output_data


def interpret_correlation(corr_value: np.float64) -> str:
    """
    Provides a simple qualitative interpretation of a correlation coefficient.

    Maps a numerical correlation value to a descriptive string (e.g.,
    "Strong positive correlation", "Weak negative correlation").

    :param corr_value: The correlation value (typically between -1.0 and 1.0).
    :returns A string describing the strength and direction of the correlation based on predefined thresholds.
    """
    # Note: These thresholds are part of the original business logic.
    if corr_value > 0.007:
        return "Strong positive correlation"
    elif corr_value > 0.3:
        return "Moderate positive correlation"
    elif corr_value > 0:
        return "Weak positive correlation"
    elif corr_value == 0:
        return "No correlation"
    elif corr_value > -0.3:
        return "Weak negative correlation"
    elif corr_value > -0.7:
        return "Moderate negative correlation"
    else:
        return "Strong negative correlation"


def calculate_correlation(
    data: dict[str, list[ScalarTag]], tags: set[str], rounding: int = 4, display_interpretation: bool = False
) -> Mapping[str, Mapping[str, Mapping[str, str | float] | float]]:
    """
    Calculate the Pearson correlation matrix for a set of tags.

    This function first aligns the scalar data by 'step'. It uses pandas to
    create a DataFrame where the index is the union of all steps, and columns
    are the tags. Values are filled with NaN for steps where a tag is missing.
    The correlation is then calculated on this aligned data.

    If two tags do not share any common steps, their correlation will be NaN
    and they will be omitted from the results.

    :param data: The input data, typically from `query_tensorboard`.
    :param tags: The set of *primary* tags to build correlations for.
        The output will be structured with these tags as the top-level keys.
    :param rounding: The number of decimal places to round the correlation value to. Defaults to 4.
    :param display_interpretation: If True, the output for each pair will
        be a dictionary {"correlation": v, "interpretation": s}.
        If False, it will just be the rounded correlation value. Defaults to False.

    :returns: A nested dictionary.
        - Top-level keys are the tags from the `tags` argument.
        - Second-level keys are the other tags being correlated against.
        - The final value is
            - A float (the rounded correlation) if `display_interpretation` is False.
            - A dict {"correlation": float, "interpretation": str} if True.
    """
    import pandas as pd

    df_data = {}
    for tag, events in data.items():
        if not events:
            continue

        tag_data = {event["step"]: event["value"] for event in events}
        df_data[tag] = tag_data

    # Create a DataFrame from the dictionary of dictionaries.
    # Pandas automatically aligns the data, using the union of all steps
    # as the index. Missing values (e.g., tag 'A' has step 5, tag 'B' doesn't)
    # are filled with NaN.
    df = pd.DataFrame(df_data)

    # Calculate the Pearson correlation matrix.
    # df.corr() handles the NaNs appropriately during calculation.
    corr_matrix = df.corr()

    # Convert the resulting pandas Correlation Matrix (a DataFrame)
    # into the specific nested dictionary format required.
    corr_dict: dict[str, dict[str, dict[str, str | float] | float]] = {}
    for i, tag1 in enumerate(corr_matrix.columns):
        # Only create top-level entries for tags in the requested `tags` set
        if tag1 not in tags:
            continue

        corr_dict[tag1] = {}
        for j, tag2 in enumerate(corr_matrix.columns):
            if i == j:
                continue  # Skip self-correlation (which is always 1.0)

            # Get the correlation value from the matrix
            corr: np.float64 = cast(np.float64, corr_matrix.iloc[i, j])

            # If correlation is NaN (e.g., no overlapping steps, or constant variance) skip this pair.
            if np.isnan(corr):
                continue

            # Round the correlation value
            corr_rounded = float(np.round(float(corr), rounding))

            # Format the output based on the display_interpretation flag
            if display_interpretation:
                corr_dict[tag1][tag2] = {"correlation": corr_rounded, "interpretation": interpret_correlation(corr)}
            else:
                corr_dict[tag1][tag2] = corr_rounded

    return corr_dict
