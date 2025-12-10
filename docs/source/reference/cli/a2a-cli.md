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

# A2A CLI Utilities

The NVIDIA NeMo Agent toolkit provides command-line utilities for testing, debugging, and interacting with A2A agents. These tools are useful for development, testing, and troubleshooting A2A integrations.

This guide covers the A2A CLI utilities. For information on using A2A clients in workflows, refer to [A2A Client](../../build-workflows/a2a-client.md). For publishing workflows as A2A servers, refer to [A2A Server](../../run-workflows/a2a-server.md).

:::{note}
**Read First**: This guide assumes familiarity with A2A concepts. Please read [A2A Client](../../build-workflows/a2a-client.md) first for foundational understanding.
:::

## Installation

A2A CLI utilities require the `nvidia-nat-a2a` package. Install it with:

```bash
uv pip install "nvidia-nat[a2a]"
```

## Available Commands

The A2A CLI provides the following commands:

- `nat a2a serve` - Start an A2A server (see [A2A Server](../../run-workflows/a2a-server.md))
- `nat a2a client discover` - Discover and inspect A2A agents
- `nat a2a client get_info` - Get agent metadata
- `nat a2a client get_skills` - List agent skills
- `nat a2a client call` - Call an A2A agent with a message

## Discovery Commands

### Start A2A Server
To run the discovery commands, you need to start an A2A server first. See [A2A Server](../../run-workflows/a2a-server.md) for more information.

```bash
nat a2a serve --config_file examples/getting_started/simple_calculator/configs/config.yml\
              --name "Calculator Agent"\
              --description "A calculator agent for mathematical operations"
```
Set server URL
```bash
export A2A_SERVER_URL=http://localhost:10000
```
### Discover Agent

The `discover` command connects to an A2A agent and displays its Agent Card, which contains information about capabilities, skills, and configuration.

**Basic usage:**
```bash
nat a2a client discover --url $A2A_SERVER_URL
```

**Output example:**
![Agent Card](../../_static/a2a_agent_card.png)


## Agent Information Commands

### Get Agent Info

Get agent metadata including name, version, provider, and capabilities.

**Usage:**
```bash
nat a2a client get_info --url $A2A_SERVER_URL
```

**Output example:**
```text
Agent Information
  Name:        Calculator Agent
  Version:     1.0.0
  URL:         http://localhost:10000/
  Description: A calculator agent for mathematical operations
  Streaming:   ✓
  Skills:      6
```

### Get Agent Skills

List all available skills with descriptions, examples, and tags.

**Usage:**
```bash
nat a2a client get_skills --url $A2A_SERVER_URL
```

**Output example:**
```text
Agent Skills (6)
  Agent: Calculator Agent

  [1] current_datetime
      Name:        Current Datetime
      Description: Returns the current date and time in human readable format with timezone information.

  [2] calculator.divide
      Name:        Calculator - Divide
      Description: Divide one number by another.

  [3] calculator.compare
      Name:        Calculator - Compare
      Description: Compare two numbers.

  [4] calculator.add
      Name:        Calculator - Add
      Description: Add two or more numbers together.

  [5] calculator.subtract
      Name:        Calculator - Subtract
      Description: Subtract one number from another.

  [6] calculator.multiply
      Name:        Calculator - Multiply
      Description: Multiply two or more numbers together.
```

## Agent Interaction Commands

### Call Agent

Call an A2A agent with a message and get a response. This is useful for quick testing and one-off queries.

**Usage:**
```bash
nat a2a client call --url $A2A_SERVER_URL --message "What is 2 + 2?"
```

**Output example:**
```text
Query: What is 2 + 2?

The sum of 2 and 2 is 4.

(0.85s)
```

## Purpose of the CLI

The A2A CLI utilities provide a quick and convenient way to interact with A2A agents from the command line. Use these commands to test agents during development, troubleshoot connection issues, explore agent capabilities, and verify that agents are working correctly before integrating them into production workflows.

## Related Documentation

- [A2A Client Guide](../../build-workflows/a2a-client.md) - Using A2A agents in workflows
- [A2A Server Guide](../../run-workflows/a2a-server.md) - Publishing workflows as A2A agents
