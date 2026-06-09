<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Tavily Search Agent Example

**Complexity:** 🟢 Beginner

This example demonstrates how to give a [`react_agent`](../../../docs/source/components/agents/react-agent/index.md) access to web search through the Tavily function group. It shows two NeMo Agent Toolkit patterns:

- Referencing a third-party function group plugin (`nemo-agent-toolkit-tavily`) by its registered `_type`.
- Routing an LLM through [LiteLLM](../../../docs/source/build-workflows/llms/index.md) so the workflow can use a non-NVIDIA model (Anthropic Claude) without changing the agent code.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Configuration](#configuration)
- [Run the Workflow](#run-the-workflow)

## Key Features

- **Tavily Function Group:** Configures the Tavily integration with `_type: tavily`. A single function group exposes multiple related tools under the `<instance_name>__<function_name>` convention, such as `tavily__search` and `tavily__extract`.
- **Tool Selection with `exclude`:** Uses `exclude: [crawl, map]` to limit the group to the search-oriented tools and keep the agent's tool list focused.
- **Function Group References:** Lists the group instance name (`tavily`) in `tool_names`. The reference auto-expands to every selected tool in the group, so you do not list each tool by name.
- **LiteLLM Routing:** Configures the LLM with `_type: litellm` and `model_name: anthropic/claude-sonnet-4-6`, which routes the request to Anthropic through LiteLLM.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit library, run the following command.

```bash
uv pip install -e examples/agents
```

### Set Up API Keys

This workflow uses the `nemo-agent-toolkit-tavily` package for web search. Create an account at [`tavily.com`](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable:

```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

The LLM is routed to Anthropic through LiteLLM, so set the `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
```

## Configuration

The agent is configured through the `configs/config.yml` file. The following configuration options are relevant to this example:

- `function_groups.tavily._type`: Set to `tavily` to load the Tavily function group from the `nemo-agent-toolkit-tavily` plugin.

- `function_groups.tavily.exclude`: A list of tools to drop from the group. This example excludes `crawl` and `map`. To restrict the group to a fixed set of tools instead, use `include` with the tool names you want.

- `llms.anthropic_llm._type`: Set to `litellm` to route the model through LiteLLM.

- `llms.anthropic_llm.model_name`: The LiteLLM model identifier. This example uses `anthropic/claude-sonnet-4-6`.

- `workflow.tool_names`: The tools the agent can call. Listing the `tavily` group instance name expands to every selected tool in the group.

- `verbose`: Defaults to `False`. When set to `True`, the agent logs input, output, and intermediate steps.

- `parse_agent_response_max_retries`: The number of times the agent retries parsing a malformed response before failing.

## Run the Workflow

Run the following command from the root of the NeMo Agent Toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file=examples/agents/tavily_search/configs/config.yml --input "What changed in the latest NVIDIA NeMo Agent Toolkit release?"
```
