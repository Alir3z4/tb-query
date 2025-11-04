# tb-query

A CLI tool and MCP (Model Context Protocol) server for querying and analyzing TensorBoard event files without requiring a running TensorBoard server.

## Overview

tb-query allows you to directly interact with TensorBoard's `events.out.tfevents.*` files to extract scalar data, calculate statistics, find correlations, and more. It's particularly useful for:

- Programmatic access to training metrics
- Automated analysis of training runs
- Integration with AI coding agents through MCP
- Quick inspection of TensorBoard logs without starting a web server

## Features

- Query scalar data with step and tag filtering
- Find all TensorBoard event files in a directory tree
- List available scalar tags with optional filtering
- Calculate statistics (min, max, mean, std) for specific tags
- Compute correlations between different scalar metrics
- CLI interface for command-line usage
- MCP server for integration with AI coding assistants

## Installation

### From PyPI

```bash
pip install tb-query
```

### From Source

```bash
git clone https://github.com/Alir3z4/tb-query.git
cd tb-query
pip install -e .
```

### Requirements

- Python >= 3.11
- tensorboard
- fastmcp
- pandas

## CLI Usage

### Query Command

Extract scalar data from a TensorBoard event file:

```bash
# Query all available tags
tb-query query path/to/events.out.tfevents.12345

# Query specific tags
tb-query query path/to/events.out.tfevents.12345 --tags loss --tags accuracy

# Query with step range filtering
tb-query query path/to/events.out.tfevents.12345 --start_step 100 --end_step 200

# Combine filters
tb-query query path/to/events.out.tfevents.12345 --tags loss --start_step 100 --end_step 200
```

Output format (JSON):
```json
{
  "loss": [
    {"step": 100, "value": 0.5},
    {"step": 101, "value": 0.48}
  ],
  "accuracy": [
    {"step": 100, "value": 0.85},
    {"step": 101, "value": 0.86}
  ]
}
```

### Tags Command

List all available scalar tags in an event file:

```bash
# List all tags
tb-query tags path/to/events.out.tfevents.12345

# Filter tags containing specific strings
tb-query tags path/to/events.out.tfevents.12345 --filter loss
tb-query tags path/to/events.out.tfevents.12345 --filter loss --filter accuracy
```

Output format (JSON):
```json
{
  "tags": ["train/loss", "train/accuracy", "eval/loss", "eval/accuracy"]
}
```

### Find Command

Locate all TensorBoard event files in a directory:

```bash
tb-query find path/to/logs
```

Output format (JSON):
```json
{
  "event_files": [
    {
      "path": "path/to/logs/run1/events.out.tfevents.12345",
      "created_at": "2025-11-04T10:30:00.123456"
    },
    {
      "path": "path/to/logs/run2/events.out.tfevents.67890",
      "created_at": "2025-11-03T15:20:00.654321"
    }
  ]
}
```

Files are sorted by creation time (newest first).

### Steps Command

Get the step numbers for specific tags:

```bash
tb-query steps path/to/events.out.tfevents.12345 --tags loss --tags accuracy
```

Output format (JSON):
```json
{
  "loss": [0, 10, 20, 30, 40, 50],
  "accuracy": [0, 10, 20, 30, 40, 50]
}
```

### Stats Command

Calculate statistical measures for tag values:

```bash
tb-query stats path/to/events.out.tfevents.12345 --tags loss --tags accuracy
```

Output format (JSON):
```json
{
  "loss": {
    "min": 0.15,
    "max": 2.34,
    "mean": 0.85,
    "std": 0.42,
    "count": 1000
  },
  "accuracy": {
    "min": 0.65,
    "max": 0.98,
    "mean": 0.87,
    "std": 0.08,
    "count": 1000
  }
}
```

### Correlation Command

Calculate Pearson correlations between scalar tags:

```bash
# Basic correlation
tb-query correlation path/to/events.out.tfevents.12345 --tags "loss,accuracy"

# With step range
tb-query correlation path/to/events.out.tfevents.12345 --tags "loss,accuracy" --start_step 100 --end_step 200

# With interpretation
tb-query correlation path/to/events.out.tfevents.12345 --tags "loss,accuracy" --display-interpretation true

# Custom rounding
tb-query correlation path/to/events.out.tfevents.12345 --tags "loss,accuracy" --rounding 6
```

Output format without interpretation (JSON):
```json
{
  "loss": {
    "accuracy": -0.9234,
    "learning_rate": 0.1234
  }
}
```

Output format with interpretation (JSON):
```json
{
  "loss": {
    "accuracy": {
      "correlation": -0.9234,
      "interpretation": "Strong negative correlation"
    }
  }
}
```

## MCP Server Usage

tb-query provides an MCP (Model Context Protocol) server that enables AI coding assistants to interact with TensorBoard event files. This allows agents to analyze training runs, extract metrics, and provide insights.

### Starting the MCP Server

```bash
tb-query-mcp
```

The server will start and listen for MCP connections from compatible clients.

### Environment Variable

You can set the `TB_QUERY_EVENTS_PATH` environment variable to specify a default directory for event files:

```bash
export TB_QUERY_EVENTS_PATH=/path/to/tensorboard/logs
tb-query-mcp
```

This enables the `event_files` resource, which automatically lists available event files from the specified directory.

### Integration with AI Coding Agents

#### Claude Desktop

Add the following configuration to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "tb-query": {
      "command": "tb-query-mcp",
      "env": {
        "TB_QUERY_EVENTS_PATH": "/path/to/your/tensorboard/logs"
      }
    }
  }
}
```

After adding the configuration, restart Claude Desktop. The tb-query tools will be available for Claude to use when analyzing your training runs.

#### Cline (VS Code Extension)

Add to your Cline MCP settings file (`.cline/mcp_settings.json` in your workspace):

```json
{
  "mcpServers": {
    "tb-query": {
      "command": "tb-query-mcp",
      "env": {
        "TB_QUERY_EVENTS_PATH": "/path/to/your/tensorboard/logs"
      }
    }
  }
}
```

#### Zed Editor

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "context_servers": {
    "tb-query": {
      "command": "tb-query-mcp",
      "env": {
        "TB_QUERY_EVENTS_PATH": "/path/to/your/tensorboard/logs"
      }
    }
  }
}
```

#### Continue (VS Code Extension)

Add to your Continue config file (`~/.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "tb-query",
      "command": "tb-query-mcp",
      "env": {
        "TB_QUERY_EVENTS_PATH": "/path/to/your/tensorboard/logs"
      }
    }
  ]
}
```

#### Using with Python Client

You can also integrate tb-query into your own Python scripts using the MCP protocol:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="tb-query-mcp",
    env={"TB_QUERY_EVENTS_PATH": "/path/to/logs"}
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Call tools
        result = await session.call_tool("list_tags", {
            "event_file": "/path/to/events.out.tfevents.12345"
        })
        print(result)
```

### Available MCP Tools

When running as an MCP server, tb-query provides the following tools:

#### query

Query scalar data from a TensorBoard event file.

Parameters:
- `event_file` (string, required): Path to the event file
- `tags` (list[string], optional): List of tags to query (default: all tags)
- `start_step` (integer, optional): Starting step (inclusive)
- `end_step` (integer, optional): Ending step (inclusive)

#### list_tags

Get all available scalar tags with optional filtering.

Parameters:
- `event_file` (string, required): Path to the event file
- `filters` (list[string], optional): Filter tags containing these strings

#### find_events

Find all TensorBoard event files in a directory and subdirectories.

Parameters:
- `directory` (string, required): Directory path to search

#### tag_steps

Get the step numbers for specified tags.

Parameters:
- `event_file` (string, required): Path to the event file
- `tags` (list[string], required): List of tags

#### tag_stats

Get statistical measures for specified tags.

Parameters:
- `event_file` (string, required): Path to the event file
- `tags` (list[string], required): List of tags

#### correlation

Calculate correlations between scalar tags.

Parameters:
- `event_file` (string, required): Path to the event file
- `tags` (list[string], required): Tags to calculate correlations for
- `start_step` (integer, optional): Starting step
- `end_step` (integer, optional): Ending step

### Available MCP Resources

#### event_files

When `TB_QUERY_EVENTS_PATH` is set, this resource provides a list of all available event files from the configured directory.

URI: `resource://event-files`

## Example Use Cases

### Monitoring Training Progress

```bash
# Check latest loss values
tb-query query events.out.tfevents.12345 --tags train/loss --start_step 990

# Compare train and validation metrics
tb-query query events.out.tfevents.12345 --tags train/loss --tags val/loss
```

### Analyzing Model Performance

```bash
# Get statistics for key metrics
tb-query stats events.out.tfevents.12345 --tags train/accuracy --tags val/accuracy

# Find correlations between metrics
tb-query correlation events.out.tfevents.12345 --tags "loss,learning_rate" --display-interpretation true
```

### AI Agent Integration

When integrated with AI coding assistants through MCP, you can simply ask:

- "Analyze the latest training run in my logs directory"
- "What's the correlation between loss and learning rate?"
- "Show me the statistics for accuracy metrics"
- "Compare the last 100 steps of train and validation loss"

The AI agent will automatically use the appropriate tb-query tools to fetch and analyze the data.

## Python API

You can also use tb-query directly in your Python code:

```python
from tb_query.core import (
    query_tensorboard,
    get_all_tags,
    find_event_files,
    get_tag_statistics,
    calculate_correlation
)

# Query data
data = query_tensorboard(
    "events.out.tfevents.12345",
    tags=["loss", "accuracy"],
    start_step=100,
    end_step=200
)

# Get tags
tags = get_all_tags("events.out.tfevents.12345", filters=["loss"])

# Get statistics
stats = get_tag_statistics("events.out.tfevents.12345", tags=["loss"])

# Calculate correlation
correlation = calculate_correlation(
    data,
    tags={"loss"},
    rounding=4,
    display_interpretation=True
)
```

### Automated Analysis Scripts

```python
import json
import subprocess

# Find all event files
result = subprocess.run(
    ["tb-query", "find", "logs/"],
    capture_output=True,
    text=True
)
event_files = json.loads(result.stdout)

# Query the most recent file
latest_file = event_files["event_files"][0]["path"]
result = subprocess.run(
    ["tb-query", "query", latest_file, "--tags", "loss"],
    capture_output=True,
    text=True
)
data = json.loads(result.stdout)

# Process the data
print(f"Final loss: {data['loss'][-1]['value']}")
```

### Using the Core Library Directly

The primary purpose of tb-query is to provide a Python library for programmatic access to TensorBoard data. All functionality is available through the `tb_query.core` module:
```python
from tb_query.core import (
    query_tensorboard,
    get_all_tags,
    find_event_files,
    get_tag_steps,
    get_tag_statistics,
    calculate_correlation,
    ValidationError
)

# Find all event files in a directory
try:
    result = find_event_files("logs/")
    event_files = result["event_files"]
    print(f"Found {len(event_files)} event files")
    
    # Use the most recent file
    latest_file = event_files[0]["path"]
    print(f"Analyzing: {latest_file}")
    
except ValidationError as e:
    print(f"Error: {e.message}")

# Get all available tags
try:
    tags_result = get_all_tags(latest_file)
    all_tags = tags_result["tags"]
    print(f"Available tags: {all_tags}")
    
    # Filter tags containing "loss"
    loss_tags = get_all_tags(latest_file, filters=["loss"])
    print(f"Loss-related tags: {loss_tags['tags']}")
    
except ValidationError as e:
    print(f"Error: {e.message}")

# Query specific tags with step filtering
try:
    data = query_tensorboard(
        event_file=latest_file,
        tags=["train/loss", "val/loss"],
        start_step=100,
        end_step=500
    )
    
    for tag, values in data.items():
        print(f"\n{tag}:")
        print(f"  First value: step={values[0]['step']}, value={values[0]['value']}")
        print(f"  Last value: step={values[-1]['step']}, value={values[-1]['value']}")
        print(f"  Total points: {len(values)}")
        
except ValidationError as e:
    print(f"Error: {e.message}")

# Get statistics for tags
try:
    stats = get_tag_statistics(latest_file, tags=["train/loss", "train/accuracy"])
    
    for tag, stat in stats.items():
        if "error" in stat:
            print(f"{tag}: {stat['error']}")
        else:
            print(f"\n{tag} statistics:")
            print(f"  Min: {stat['min']:.4f}")
            print(f"  Max: {stat['max']:.4f}")
            print(f"  Mean: {stat['mean']:.4f}")
            print(f"  Std: {stat['std']:.4f}")
            print(f"  Count: {stat['count']}")
            
except ValidationError as e:
    print(f"Error: {e.message}")

# Get available steps for specific tags
try:
    steps = get_tag_steps(latest_file, tags=["train/loss", "val/loss"])
    
    for tag, step_list in steps.items():
        print(f"{tag}: {len(step_list)} steps")
        print(f"  Range: {step_list[0]} to {step_list[-1]}")
        
except ValidationError as e:
    print(f"Error: {e.message}")

# Calculate correlations
try:
    # First query the data
    data = query_tensorboard(
        event_file=latest_file,
        tags=None,  # Get all tags
        start_step=0,
        end_step=1000
    )
    
    # Calculate correlation for specific tags
    correlation = calculate_correlation(
        data=data,
        tags={"train/loss"},  # Primary tag(s) to correlate against others
        rounding=4,
        display_interpretation=False
    )
    
    print("\nCorrelations with train/loss:")
    for other_tag, corr_value in correlation["train/loss"].items():
        print(f"  {other_tag}: {corr_value}")
    
    # With interpretation
    correlation_interpreted = calculate_correlation(
        data=data,
        tags={"train/loss"},
        rounding=4,
        display_interpretation=True
    )
    
    print("\nCorrelations with interpretation:")
    for other_tag, corr_data in correlation_interpreted["train/loss"].items():
        print(f"  {other_tag}:")
        print(f"    Correlation: {corr_data['correlation']}")
        print(f"    Interpretation: {corr_data['interpretation']}")
        
except ValidationError as e:
    print(f"Error: {e.message}")

# Complete analysis workflow
def analyze_training_run(event_file_path: str):
    """Complete analysis of a training run."""
    try:
        # Get all tags
        tags_result = get_all_tags(event_file_path)
        all_tags = tags_result["tags"]
        
        # Get statistics for all tags
        stats = get_tag_statistics(event_file_path, tags=all_tags)
        
        # Query recent data (last 100 steps)
        data = query_tensorboard(event_file_path, tags=all_tags)
        
        # Get the maximum step across all tags
        max_step = 0
        for tag_data in data.values():
            if tag_data:
                max_step = max(max_step, tag_data[-1]["step"])
        
        # Query only recent data
        recent_data = query_tensorboard(
            event_file_path,
            tags=all_tags,
            start_step=max(0, max_step - 100),
            end_step=max_step
        )
        
        # Calculate correlations
        correlation = calculate_correlation(
            data=data,
            tags=set(all_tags[:5]),  # Limit to first 5 tags to avoid huge output
            rounding=4,
            display_interpretation=True
        )
        
        return {
            "tags": all_tags,
            "statistics": stats,
            "recent_data": recent_data,
            "correlations": correlation,
            "max_step": max_step
        }
        
    except ValidationError as e:
        return {"error": e.message}

# Use the analysis function
result = analyze_training_run("events.out.tfevents.12345")
if "error" in result:
    print(f"Analysis failed: {result['error']}")
else:
    print(f"Analysis complete: {len(result['tags'])} tags analyzed")
    print(f"Training ran for {result['max_step']} steps")
```

All functions in `tb_query.core` raise `ValidationError` exceptions for file access or parsing errors, so wrapping calls in try-except blocks is recommended for robust error handling.


## Error Handling

tb-query provides clear error messages for common issues:

- **File not found**: Raised when the specified event file doesn't exist
- **Failed to load event file**: Raised when the file is corrupted or invalid
- **Directory not found**: Raised when the specified directory doesn't exist
- **Tag not found**: Returned in statistics when a requested tag doesn't exist
- **No values found**: Returned when a tag exists but has no data points

## Development

### Setup Development Environment

```bash
git clone https://github.com/Alir3z4/tb-query.git
cd tb-query
make install
```

### Run Tests

_Currently, the code base doesn't include tests and I plan to add them later._

#### Running the tests

```bash
make test
```

#### Running the tests with coverage
```bash
# Run tests with coverage
make coverage
coverage report
```

### Code Quality

```bash
make lint
```

#### Pre Commit

There is a makefile task that runs the formatting and type checking. To be used before commiting the code.

```bash
make precommit
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GPL-3.0-or-later License - see the LICENSE file for details.

## Links

- **Homepage**: https://github.com/Alir3z4/tb-query
- **Repository**: https://github.com/Alir3z4/tb-query.git
- **Issues**: https://github.com/Alir3z4/tb-query/issues
- **Changelog**: https://github.com/Alir3z4/tb-query/blob/master/ChangeLog.md

## Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.
