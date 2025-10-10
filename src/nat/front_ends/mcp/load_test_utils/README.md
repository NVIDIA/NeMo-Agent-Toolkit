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

Run the example load test:

```bash
python src/nat/front_ends/mcp/load_test_utils/example_load_test.py
```

## Usage

### Basic Example

```python
from nat.front_ends.mcp.load_test_utils import run_load_test

results = run_load_test(
    config_file="examples/getting_started/simple_calculator/configs/config.yml",
    tool_calls=[
        {"tool_name": "calculator_multiply", "args": {"text": "2 * 3"}},
        {"tool_name": "calculator_divide", "args": {"text": "10 / 2"}},
    ],
    num_concurrent_users=10,
    duration_seconds=30,
)
```

### Advanced Configuration

```python
from nat.front_ends.mcp.load_test_utils import run_load_test

results = run_load_test(
    config_file="path/to/config.yml",
    tool_calls=[
        {
            "tool_name": "calculator_multiply",
            "args": {"text": "2 * 3"},
            "weight": 2.0,  # Called twice as often
        },
        {
            "tool_name": "calculator_divide",
            "args": {"text": "10 / 2"},
            "weight": 1.0,
        },
    ],
    num_concurrent_users=50,
    duration_seconds=120,
    server_port=9901,
    transport="streamable-http",
    warmup_seconds=10,
    output_dir="my_load_test_results",
)
```

## Configuration Parameters

- `config_file` (str): Path to NAT workflow config file
- `tool_calls` (list): List of tool configurations with:
  - `tool_name` (str): MCP tool name
  - `args` (dict): Tool arguments
  - `weight` (float, optional): Relative call frequency (default: 1.0)
- `num_concurrent_users` (int): Concurrent user count (default: 10)
- `duration_seconds` (int): Test duration (default: 60)
- `server_host` (str): Server host (default: "localhost")
- `server_port` (int): Server port (default: 9901)
- `transport` (str): Transport type: "streamable-http" or "sse" (default: "streamable-http")
- `warmup_seconds` (int): Warmup period before measurements (default: 5)
- `output_dir` (str): Report output directory (default: "load_test_results")

## Output

The load test generates three report files in the output directory:

1. **JSON Report** (`load_test_YYYYMMDD_HHMMSS.json`): Complete results and statistics
2. **HTML Report** (`load_test_YYYYMMDD_HHMMSS.html`): Visual dashboard with charts
3. **CSV Report** (`load_test_YYYYMMDD_HHMMSS.csv`): Raw result data for analysis

### Report Contents

- **Summary Metrics**: Total requests, success rate, requests per second
- **Latency Statistics**: Mean, median, P95, P99, min, max latencies
- **Per-Tool Statistics**: Individual performance for each tool
- **Error Analysis**: Failed request breakdown

## Customizing Your Load Test

1. Copy `example_load_test.py` to your preferred location
2. Update `config_file` to point to your NAT workflow configuration
3. Modify `tool_calls` to match your MCP tools and arguments
4. Adjust load test parameters as needed
5. Run the script: `python your_load_test.py`

## Requirements

- NeMo Agent toolkit with MCP support (`nvidia-nat[mcp]`)
- Valid NAT workflow configuration with MCP-compatible tools
- Python 3.10 or higher
