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

# Agent Examples

The agent examples demonstrate how NeMo Agent toolkit accelerates and enables AI Agent development.
The examples showcase 5 distinct AI Agent architectures solving a similar problem in different ways.
By leveraging the NeMo Agent toolkit plugin system and the `Builder` object, you can use both pre-built and
custom agentic workflows and tools in a flexible manner.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
  - [Set Up API Keys](#set-up-api-keys)
- [Example Usage](#example-usage)
- [Learn More](#learn-more)

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Set Up API Keys

If you have not already done so, follow the [Obtaining API Keys](../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

Each agent example contains its own installation and usage instructions. Navigate to the specific example directory and follow the README instructions:

- **ReAct Agent**: See [react/README.md](./react/README.md) for step-by-step reasoning agent implementation
- **Tool Calling Agent**: See [tool_calling/README.md](./tool_calling/README.md) for direct tool invocation agent
- **Mixture of Agents**: See [mixture_of_agents/README.md](./mixture_of_agents/README.md) for multi-agent orchestration
- **ReWOO Agent**: See [rewoo/README.md](./rewoo/README.md) for planning-based agent workflow

## Learn More

For a deeper dive into the AI Agents utilized in the examples, refer to the component documentation:
- [ReAct Agent](../../docs/source/workflows/about/react-agent.md)
- [Reasoning Agent](../../docs/source/workflows/about/reasoning-agent.md)
- [Tool Calling Agent](../../docs/source/workflows/about/tool-calling-agent.md)
- [ReWOO Agent](../../docs/source/workflows/about/rewoo-agent.md)
