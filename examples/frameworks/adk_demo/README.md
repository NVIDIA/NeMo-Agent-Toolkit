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
<!-- path-check-skip-file -->
# Google Agent Development Kit (ADK) Example

A minimal example using Agent Development Kit showcasing a simple weather time agent that can call tools (a function tool and an MCP tool).

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NAT library, run the following command:

```bash
uv pip install -e examples/frameworks/adk_demo
```

### Set up API keys

LiteLLM routes to many providers. For OpenAI:
```bash
export OPENAI_API_KEY="<your_openai_key>"
# Optional (defaults to https://api.openai.com/v1 if unset)
export OPENAI_API_BASE="<your_openai_base_url>"
```
For Azure OpenAI, set:
```bash
export OPENAI_API_KEY="<your_azure_openai_key>"
export OPENAI_API_BASE="https://<your-azure-endpoint>/openai"
```
You can find LLM provider specific instructions in the LiteLLM documentation. Please set the appropriate environment variables.

### Set up the MCP Server

This example also demonstrates how NAT can interact with MCP servers on behalf of ADK.

First run the MCP server with this command.

```bash
nat mcp serve --config_file examples/frameworks/adk_demo/configs/config.yml --host 0.0.0.0 --port 9901 --name "My MCP Server"
```

## Run the Workflow

Then run the workflow with the NAT CLI

```bash
nat run --config_file examples/frameworks/adk_demo/configs/config.yml --input "What is the weather and time in New York today?"
```

### Expected Output

```console
<snipped for brevity>

Configuration Summary:
--------------------
Workflow Type: adk
Number of Functions: 2
Number of Function Groups: 0
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0

<snipped for brevity>

--------------------------------------------------
Workflow Result:
['Here’s the latest for New York:\n- Weather: Sunny, around 25°C (77°F)\n- Time: 2025-09-25 12:27:26 EDT (UTC-4)']
--------------------------------------------------
```
