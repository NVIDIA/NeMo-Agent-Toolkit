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

# ReWOO Agent Example

This example demonstrates how to use A configurable [ReWOO](https://arxiv.org/abs/2305.18323) (Reasoning WithOut Observation) Agent with the NeMo Agent toolkit. For this purpose NeMo Agent toolkit provides a [`rewoo_agent`](../../../docs/source/workflows/about/rewoo-agent.md) workflow type.

## Table of Contents

- [Key Features](#key-features)
- [Graph Structure](#graph-structure)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)
  - [Starting the NeMo Agent Toolkit Server](#starting-the-nemo-agent-toolkit-server)
  - [Making Requests to the NeMo Agent Toolkit Server](#making-requests-to-the-nemo-agent-toolkit-server)
  - [Evaluating the ReWOO Agent Workflow](#evaluating-the-rewoo-agent-workflow)

## Key Features

- **ReWOO Agent Architecture:** Demonstrates the unique `rewoo_agent` workflow type that implements Reasoning Without Observation, separating planning, execution, and solving into distinct phases.
- **Three-Node Graph Structure:** Uses a distinctive architecture with Planner Node (creates complete execution plan), Executor Node (executes tools systematically), and Solver Node (synthesizes final results).
- **Systematic Tool Execution:** Shows how ReWOO first plans all necessary steps upfront, then executes them systematically without dynamic re-planning, leading to more predictable tool usage patterns.
- **Calculator and Internet Search Integration:** Includes `calculator_inequality` and `internet_search` tools to demonstrate multi-step reasoning that requires both mathematical computation and web research.
- **Plan-Execute-Solve Pattern:** Demonstrates the ReWOO approach of complete upfront planning followed by systematic execution and final result synthesis.

## Graph Structure

The ReWOO agent uses a unique three-node graph architecture that separates planning, execution, and solving into distinct phases. The following diagram illustrates the agent's workflow:

<div align="center">
<img src="../../../docs/source/_static/rewoo_agent.png" alt="ReWOO Agent Graph Structure" width="400" style="max-width: 100%; height: auto;">
</div>

**Workflow Overview:**
- **Start**: The agent begins processing with user input
- **Planner Node**: Creates a complete execution plan with all necessary steps upfront
- **Executor Node**: Executes tools according to the plan, looping until all steps are completed
- **Solver Node**: Takes all execution results and generates the final answer
- **End**: Process completes with the final response

This architecture differs from other agents by separating reasoning (planning) from execution, allowing for more systematic and predictable tool usage patterns. The ReWOO approach first plans all steps, then executes them systematically, and finally synthesizes the results.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv sync --all-groups --all-extras
uv pip install -e .
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com``](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

## Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file=examples/agents/rewoo/configs/config.yml --input "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"
```

**Expected Workflow Result**
```
Athens
```
---

### Starting the NeMo Agent Toolkit Server

You can start the NeMo Agent toolkit server using the `aiq serve` command with the appropriate configuration file.

**Starting the ReWOO Agent Example Workflow**

```bash
aiq serve --config_file=examples/agents/rewoo/configs/config.yml
```

### Making Requests to the NeMo Agent Toolkit Server

Once the server is running, you can make HTTP requests to interact with the workflow.

#### Non-Streaming Requests

**Non-Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"}'
```

#### Streaming Requests

**Streaming Request to the ReWOO Agent Example Workflow**

```bash
curl --request POST \
  --url http://localhost:8000/generate/stream \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Which city held the Olympic game in the year represented by the bigger number of 1996 and 2004?"}'
```
---

### Evaluating the ReWOO Agent Workflow
**Run and evaluate the `rewoo_agent` example Workflow**

```bash
aiq eval --config_file=examples/agents/rewoo/configs/config.yml
```
