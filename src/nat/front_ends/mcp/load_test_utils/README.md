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

Load testing utilities for NeMo Agent toolkit MCP servers. Simulate concurrent users making tool calls and generate detailed performance reports.

## Quick Start

Run a load test using the main entry point script:

```bash
# From the load_test_utils directory
python cli.py --config_file=configs/config.yml

# Or from the project root
python src/nat/front_ends/mcp/load_test_utils/cli.py \
  --config_file=src/nat/front_ends/mcp/load_test_utils/configs/config.yml
```

List available configurations:

```bash
python cli.py --list-configs
```

Get help:

```bash
python cli.py --help
```

## Configuration

All load test options are configured using YAML files stored in the `configs/` directory.

### Example Configuration

```yaml
# Path to NAT workflow config file
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

# Tool calls to execute
tool_calls:
  - tool_name: "calculator_multiply"
    args:
      text: "2 * 3"
    weight: 2.0  # Called twice as often

  - tool_name: "calculator_divide"
    args:
      text: "10 / 2"
    weight: 1.0
```

### Configuration Options

#### Required Fields

- `config_file` (str): Path to NAT workflow configuration file

#### Server Configuration (`server`)

- `host` (str): Server host (default: "localhost")
- `port` (int): Server port (default: 9901)
- `transport` (str): Transport type - "streamable-http" or "sse" (default: "streamable-http")

#### Load Test Parameters (`load_test`)

- `num_concurrent_users` (int): Number of concurrent users to simulate (default: 10)
- `duration_seconds` (int): Test duration in seconds (default: 60)
- `warmup_seconds` (int): Warmup period before measurements (default: 5)

#### Output Configuration (`output`)

- `directory` (str): Output directory for reports (default: "load_test_results")

#### Tool Calls (`tool_calls`)

List of tool configurations:
- `tool_name` (str, required): Name of the MCP tool
- `args` (dict, optional): Arguments to pass to the tool
- `weight` (float, optional): Relative call frequency (default: 1.0)

Higher weight values mean the tool will be called more frequently. For example, a tool with weight 2.0 will be called twice as often as a tool with weight 1.0.

## Usage

### Command Line (Recommended)

Run load tests using the main entry point script:

```bash
# Basic usage
python cli.py --config_file=configs/config.yml

# With verbose logging
python cli.py --config_file=configs/config.yml --verbose

# Short form
python cli.py -c configs/config.yml
```

### Python API

#### Using YAML Configuration

```python
from nat.front_ends.mcp.load_test_utils import run_load_test_from_yaml

results = run_load_test_from_yaml("configs/config.yml")
```

#### Programmatic Usage (Advanced)

```python
from nat.front_ends.mcp.load_test_utils import run_load_test

results = run_load_test(
    config_file="examples/getting_started/simple_calculator/configs/config.yml",
    tool_calls=[
        {"tool_name": "calculator_multiply", "args": {"text": "2 * 3"}, "weight": 2.0},
        {"tool_name": "calculator_divide", "args": {"text": "10 / 2"}, "weight": 1.0},
    ],
    num_concurrent_users=10,
    duration_seconds=30,
)
```

## Output

The load test generates two report files in the output directory:

1. **CSV Report** (`load_test_YYYYMMDD_HHMMSS.csv`): Detailed per-request data with columns:
   - `timestamp`: Request timestamp
   - `tool_name`: Name of the tool called
   - `success`: Boolean success status
   - `latency_ms`: Request latency in milliseconds
   - `memory_rss_mb`: Resident Set Size (RSS) memory in MB at request time
   - `memory_vms_mb`: Virtual Memory Size (VMS) in MB at request time
   - `memory_percent`: Memory usage percentage at request time
   - `error`: Error message (if failed)

2. **Summary Report** (`load_test_YYYYMMDD_HHMMSS_summary.txt`): Human-readable summary with statistics

### Report Contents

- **Summary Metrics**: Total requests, success rate, requests per second
- **Latency Statistics**: Mean, median, P95, P99, min, max latencies
- **Memory Statistics**: RSS and VMS memory usage (mean, max), percentage (mean, max)
- **Per-Tool Statistics**: Individual performance for each tool
- **Error Analysis**: Failed request breakdown

## Creating Custom Tests

1. Copy the config file from the `configs/` directory:
   ```bash
   cp configs/config.yml configs/my_test.yml
   ```

2. Modify the parameters in `configs/my_test.yml` for your use case:
   - Update `config_file` to point to your NAT workflow
   - Adjust `tool_calls` to match your available tools
   - Set load test parameters (`num_concurrent_users`, `duration_seconds`)

3. Run your custom test:
   ```bash
   python cli.py --config_file=configs/my_test.yml
   ```

## Requirements

- NeMo Agent toolkit with MCP support (`nvidia-nat[mcp]`)
- Valid NAT workflow configuration with MCP-compatible tools
- Python 3.10 or higher
- psutil package (for memory monitoring)
