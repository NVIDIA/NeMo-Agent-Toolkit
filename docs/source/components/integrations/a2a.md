<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Agent-to-Agent Protocol (A2A)

NVIDIA NeMo Agent toolkit [Agent-to-Agent Protocol (A2A)](https://a2aproject.org/) integration includes:
* An [A2A client](../../build-workflows/a2a-client.md) to connect to and interact with remote A2A [agents](../agents/index.md).
* An [A2A server](../../run-workflows/a2a-server.md) to publish [workflows](../../build-workflows/about-building-workflows.md) as A2A agents that can be discovered and invoked by other A2A clients.

**Note:** A2A functionality requires the `nvidia-nat-a2a` package. Install it with `uv pip install "nvidia-nat[a2a]"`.

## What is A2A?

The Agent-to-Agent (A2A) Protocol is an open standard from the Linux Foundation that enables agent-to-agent communication and collaboration. A2A standardizes how agents:
- **Discover capabilities** through Agent Cards
- **Delegate tasks** to other agents
- **Exchange information** using a common protocol

## Key Concepts

### A2A Agent
A service that exposes capabilities (skills) via the A2A protocol. Agents publish an Agent Card describing their capabilities and accept task requests from clients.

### Agent Card
JSON metadata describing an A2A agent's capabilities, including:
- Agent name, version, and description
- Available skills with descriptions and examples
- Supported capabilities (streaming, push notifications)
- Content types (input/output modes)

### A2A Client
A component that connects to remote A2A agents and invokes their skills. The `a2a_client` [function group](../../build-workflows/functions-and-function-groups/function-groups.md) provides a function interface for interacting with remote agents.

### A2A Server
A service that exposes workflows as A2A agents. The `nat a2a serve` command publishes workflows so they can be discovered and called by other A2A clients.

## Examples

The following examples demonstrate A2A integration:

- **Math Assistant A2A** (`examples/A2A/math_assistant_a2a/`) - A2A communication with hybrid [tool](../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) composition (A2A calculator + MCP time + local logic)
- **Currency Agent A2A** (`examples/A2A/currency_agent_a2a/`) - Connecting to external third-party A2A services (LangGraph-based currency agent)

## Documentation

- [Connecting to Remote Agents](../../build-workflows/a2a-client.md)
- [Publishing Workflows](../../run-workflows/a2a-server.md)
- [A2A Authentication](../auth/a2a-auth.md)

## Protocol Compliance

The A2A integration is built on the official [A2A Python SDK](https://github.com/a2aproject/a2a-python) to ensure protocol compliance. For detailed protocol specifications, refer to the [A2A Protocol Documentation](https://a2a-protocol.org/latest/specification/).

## A2A vs MCP

Both A2A and MCP enable integration with external capabilities, but they serve different purposes:

| Aspect | A2A | MCP |
|--------|-----|-----|
| **Purpose** | Agent-to-agent communication | Tool and context integration |
| **Granularity** | Agent level (high-level tasks) | Tool level (specific functions) |
| **Discovery** | Agent Card with skills | Tool list with schemas |
| **Use Case** | Delegating to other agents | Accessing tools and context |
| **Best For** | Multi-agent systems | Tool integration |

You typically use A2A to delegate complex tasks to other agents and MCP to access tools and context. You can use both protocols together for maximum flexibility.
