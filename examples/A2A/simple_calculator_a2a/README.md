<!-- SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# A2A Calculator Client Example

This example demonstrates a sophisticated A2A client that connects to a NAT-based calculator server while integrating with local tools, showcasing end-to-end NAT-to-NAT A2A communication with hybrid tool composition.

## Key Features

- **A2A Protocol Integration**: Connects to a remote NAT calculator workflow via A2A protocol
- **Hybrid Tool Architecture**: Combines remote A2A tools with local MCP and custom functions
- **Multi-step Reasoning**: Demonstrates complex problem-solving requiring coordination between different tool types

## Architecture Overview

```
┌─────────────────┐    A2A Protocol    ┌─────────────────┐
│   Calculator    │◄──────────────────►│   Calculator    │
│   Client        │  (localhost:10000) │   Server        │
│   Workflow      │                    │   (NAT-based)   │
│                 │                    │                 │
│ Tools:          │                    │ Tools:          │
│ • A2A Calculator│                    │ • calculator.*  │
│ • MCP Time      │                    │ • current_time  │
│ • Format Utils  │                    └─────────────────┘
└─────────────────┘
```

## Installation and Setup

### Prerequisites

Follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install Dependencies

From the root directory of the NeMo Agent toolkit library, install this example:

```bash
uv pip install -e examples/A2A/simple_calculator_a2a
```

### Set Up API Keys

Set your NVIDIA API key as an environment variable:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Usage

### Start the Calculator A2A Server

First, start the calculator server that this client will connect to:

```bash
# Terminal 1: Start the A2A calculator server
nat a2a serve --config_file examples/getting_started/simple_calculator/configs/config.yml --port 10000
```

### Run the Calculator Client

In a separate terminal, run the client workflow:

```bash
# Terminal 2: Run the calculator client
nat run --config_file examples/A2A/simple_calculator_a2a/configs/config.yml \
  --input "Calculate the compound interest on $1000 at 5% annual interest for 3 years, then tell me if the final amount is greater than 1200"
```

## Example Queries

The agent demonstrates hybrid tool capabilities, coordinating between A2A calculator functions, MCP time services, and local utility functions.

### Quick Example

```bash
# Basic hybrid calculation combining math and time
nat run --config_file examples/A2A/simple_calculator_a2a/configs/config.yml \
  --input "Is the product of 15 and 8 greater than the current hour of the day?"
```

### Additional Examples

For comprehensive examples demonstrating different capabilities (basic calculations, time-integrated math, multi-step problems, business applications, and advanced reasoning), see [`data/sample_queries.json`](data/sample_queries.json).

You can run any example from the dataset:

```bash
# Using jq to extract a specific query
nat run --config_file examples/A2A/simple_calculator_a2a/configs/config.yml \
  --input "$(jq -r '.[] | select(.id == \"business_01\") | .query' examples/A2A/simple_calculator_a2a/data/sample_queries.json)"
```

## Configuration Details

### Tool Composition

The configuration demonstrates three types of tool integration:

1. **A2A Client Tools** (`calculator_a2a`):
   - Connects to remote calculator server
   - Provides: `add`, `subtract`, `multiply`, `divide`, `compare` functions
   - Timeout: 60 seconds
   - Skill descriptions included for better agent understanding

2. **MCP Client Tools** (`mcp_time`):
   - Local MCP server for time operations
   - Provides: `get_current_time_mcp` function
   - Configured for Pacific timezone

3. **Custom Functions** (`format_number`):
   - Local utility for number formatting
   - Provides: `format_number` with decimal precision control

### Workflow Configuration

- **Agent Type**: ReAct agent for multi-step reasoning
- **LLM**: NVIDIA NIM with deterministic responses (temperature: 0.0)
- **Error Handling**: Retry logic with 3 max retries
- **Verbose Mode**: Enabled for transparency in tool usage

## Comparison with Other A2A Examples

| Feature | Simple Calculator A2A | Currency Agent A2A |
|---------|----------------------|-------------------|
| Server Type | NAT Workflow | External Service |
| Tool Integration | Hybrid (A2A + MCP + Local) | Single A2A Client |
| Use Case | End-to-end NAT demo | External integration |
| Complexity | Advanced orchestration | Basic connectivity |

## Troubleshooting

### Connection Issues

**Server Not Running**:
```bash
# Check if server is running
curl http://localhost:10000/.well-known/agent-card.json
```

**Port Conflicts**:
- Ensure port 10000 is available for the calculator server
- Check for other services using the port

### Tool Discovery Issues

**MCP Server Not Found**:
```bash
# Verify MCP time server installation
python -m mcp_server_time --help
```

### Performance Issues

**Timeouts**:
- Increase `task_timeout` in config if calculations take longer
- Check network connectivity to remote services

## Advanced Configuration

### Custom Time Zones

Modify the MCP time server configuration:

```yaml
mcp_time:
  _type: mcp_client
  server:
    transport: stdio
    command: "python"
    args: ["-m", "mcp_server_time", "--local-timezone=Europe/London"]
```

### Multiple A2A Servers

Extend configuration to connect to multiple A2A servers:

```yaml
function_groups:
  calculator_a2a:
    _type: a2a_client
    url: http://localhost:10000

  another_service:
    _type: a2a_client
    url: http://localhost:10001
```

## Development and Testing

### Running Tests

```bash
# From the example directory
python -m pytest tests/ -v
```

### Testing with Example Dataset

The [`data/sample_queries.json`](data/sample_queries.json) file contains structured test queries in a format compatible with `nat eval`. You can use these examples for both manual testing and automated evaluation.

#### Manual Testing

```bash
# Test all examples in the dataset
for query_id in $(jq -r '.[].id' examples/A2A/simple_calculator_a2a/data/sample_queries.json); do
  echo "Testing question $query_id..."
  query=$(jq -r ".[] | select(.id == $query_id) | .question" examples/A2A/simple_calculator_a2a/data/sample_queries.json)
  nat run --config_file examples/A2A/simple_calculator_a2a/configs/config.yml --input "$query"
done
```

Or test specific categories:

```bash
# Test only basic calculations
jq -r '.[] | select(.category == "basic_calculations") | .question' examples/A2A/simple_calculator_a2a/data/sample_queries.json | \
while read -r query; do
  nat run --config_file examples/A2A/simple_calculator_a2a/configs/config.yml --input "$query"
done
```

#### Automated Evaluation

You can also use the dataset for automated evaluation with `nat eval`:

```bash
# Create an evaluation config file
cat > eval_config.yml << EOF
function_groups:
  calculator_a2a:
    _type: a2a_client
    url: http://localhost:10000
  mcp_time:
    _type: mcp_client
    server:
      transport: stdio
      command: "python"
      args: ["-m", "mcp_server_time"]
  format_number:
    _type: nat_function

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 1024

workflow:
  _type: react_agent
  tool_names: [calculator_a2a, mcp_time, format_number]
  llm_name: nim_llm
  verbose: true

eval:
  general:
    max_concurrency: 1
    output_dir: .tmp/nat/examples/A2A/simple_calculator_a2a
    dataset:
      _type: json
      file_path: examples/A2A/simple_calculator_a2a/data/sample_queries.json
  evaluators:
    basic_eval:
      _type: basic_evaluator
EOF

# Run evaluation
nat eval --config_file eval_config.yml
```

### Debugging

Enable detailed logging:

```bash
export NAT_LOG_LEVEL=DEBUG
nat run --config_file examples/A2A/simple_calculator_a2a/configs/config.yml --input "Test query"
```

## Contributing

When modifying this example:

1. Test both server and client workflows
2. Update documentation for new features
3. Ensure backward compatibility
4. Add example queries for new capabilities

## Related Examples

- [Simple Calculator](../../../../examples/getting_started/simple_calculator/) - The A2A server this client connects to
- [Currency Agent A2A](../currency_agent_a2a/) - External A2A service integration example
