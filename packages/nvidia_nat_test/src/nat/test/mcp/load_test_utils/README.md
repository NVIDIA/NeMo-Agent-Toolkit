<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# MCP Server Load Testing

This utility simulates concurrent users making tool calls to MCP servers and generates detailed performance reports for NVIDIA NeMo Agent toolkit.

## Requirements

Before running load tests, ensure you have the following:

- NeMo Agent toolkit with MCP support installed through `nvidia-nat[mcp]`
- NeMo Agent toolkit with Test support installed through `nvidia-nat[test]`
- Valid NeMo Agent toolkit workflow configuration with MCP-compatible tools

The `psutil` package is required for monitoring server memory usage during load tests. Install it using the following command:

```bash
uv pip install psutil
```

## Quick Start

Run a load test from the project root:

```bash
python packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/cli.py \
  --config_file=packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/configs/config.yml
```

Get help:

```bash
python packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/cli.py --help
```

## Configuration

Configure load test options using YAML files stored in the `configs/` directory.

### Example Configuration

```yaml
# Path to NeMo Agent toolkit workflow configuration file
config_file: "examples/getting_started/simple_calculator/configs/config.yml"

# Server configuration
server:
  host: "localhost"
  port: 9901
  transport: "streamable-http"  # Options: "streamable-http" or "sse"

# Load test parameters
load_test:
  num_concurrent_users: 10
  duration_seconds: 30
  warmup_seconds: 5

# Output configuration
output:
  directory: "load_test_results"

# Tool calls to execute during load testing
tool_calls:
  - tool_name: "calculator_multiply"
    args:
      text: "2 * 3"
    weight: 2.0  # Called twice as often as weight 1.0 tools

  - tool_name: "calculator_divide"
    args:
      text: "10 / 2"
    weight: 1.0
```

### Configuration Parameters

#### Required Parameters

**`config_file`** (string)
: Path to the NeMo Agent toolkit workflow configuration file.

#### Server Configuration

Configure the MCP server settings in the `server` section:

**`host`** (string, default: `"localhost"`)
: Host address where the MCP server will run.

**`port`** (integer, default: `9901`)
: Port number for the MCP server.

**`transport`** (string, default: `"streamable-http"`)
: Transport protocol type. Options: `"streamable-http"` or `"sse"`.

#### Load Test Parameters

Configure load test behavior in the `load_test` section:

**`num_concurrent_users`** (integer, default: `10`)
: Number of concurrent users to simulate.

**`duration_seconds`** (integer, default: `60`)
: Duration of the load test in seconds.

**`warmup_seconds`** (integer, default: `5`)
: Warmup period before measurements begin, in seconds.

#### Output Configuration

Configure report output in the `output` section:

**`directory`** (string, default: `"load_test_results"`)
: Directory where test reports will be saved.

#### Tool Calls

Define tool calls to execute in the `tool_calls` list. Each tool call includes:

**`tool_name`** (string, required)
: Name of the MCP tool to call.

**`args`** (dictionary, optional)
: Arguments to pass to the tool.

**`weight`** (float, default: `1.0`)
: Relative call frequency. Tools with higher weights are called more frequently. A tool with weight 2.0 is called twice as often as a tool with weight 1.0.

## Running Load Tests

### Command Line

Run load tests from the project root using the command-line interface:

```bash
# Basic usage
python packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/cli.py \
  --config_file=packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/configs/config.yml

# With verbose logging
python packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/cli.py \
  --config_file=packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/configs/config.yml \
  --verbose

# Short form
python packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/cli.py \
  -c packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/configs/config.yml
```

### Python API

#### Using YAML Configuration

```python
from nat.test.mcp.load_test_utils import run_load_test_from_yaml

results = run_load_test_from_yaml(
    "packages/nvidia_nat_test/src/nat/test/mcp/load_test_utils/configs/config.yml"
)
```

#### Programmatic Usage

```python
from nat.test.mcp.load_test_utils import run_load_test

results = run_load_test(
    config_file="examples/getting_started/simple_calculator/configs/config.yml",
    tool_calls=[
        {
            "tool_name": "calculator_multiply",
            "args": {"text": "2 * 3"},
            "weight": 2.0,
        },
        {
            "tool_name": "calculator_divide",
            "args": {"text": "10 / 2"},
            "weight": 1.0,
        },
    ],
    num_concurrent_users=10,
    duration_seconds=30,
)
```

## Output Reports

The load test generates two report files in the output directory:

### CSV Report

**File name**: `load_test_YYYYMMDD_HHMMSS.csv`

Detailed per-request data with the following columns:

- `timestamp`: Request timestamp
- `tool_name`: Name of the tool called
- `success`: Boolean success status
- `latency_ms`: Request latency in milliseconds
- `memory_rss_mb`: Resident Set Size (RSS) memory in MB at request time
- `memory_vms_mb`: Virtual Memory Size (VMS) in MB at request time
- `memory_percent`: Memory usage percentage at request time
- `error`: Error message if the request failed

### Summary Report

**File name**: `load_test_YYYYMMDD_HHMMSS_summary.txt`

Human-readable summary with the following statistics:

**Summary Metrics**
: Total requests, success rate, requests per second

**Latency Statistics**
: Mean, median, P95, P99, minimum, and maximum latencies

**Memory Statistics**
: RSS and VMS memory usage (mean and max), memory percentage (mean and max)

**Per-Tool Statistics**
: Individual performance metrics for each tool

**Error Analysis**
: Breakdown of failed requests by error type
