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
# Configure the Responses API and Agent
Configure the Responses API and [agent](../index.md) with a [LLM](../../../build-workflows/llms/index.md) client and through the agent graph.

## Requirements

- A model that supports the Responses API and any enabled built-in tools.
- For MCP usage, a reachable MCP server and any necessary credentials.


## LLM Configuration: `api_type`

LLM clients support an `api_type` selector. By default, `api_type` is `chat_completions`. To use the Responses API, set `api_type` to `responses` in your LLM configuration.

### Example LLM Configuration

```yaml
llms:
  openai_llm:
    _type: openai
    model_name: gpt-5-mini-2025-08-07
    # Default is `chat_completions`; set to `responses` to enable the Responses API
    api_type: responses
```

:::{note}
- If the `api_type` is omitted, the client will use `chat_completions`.
- The Responses API unlocks built-in tools and MCP integration.
:::

## Agent Configuration: `_type: responses_api_agent`

The Responses API agent binds tools directly to the LLM for execution under the Responses API, while NeMo Agent toolkit tools run through the agent graph. This preserves the familiar flow of the NeMo Agent toolkit with added tool capabilities.

### Example `config.yml`

```yaml
functions:
  current_datetime:
    _type: current_datetime

llms:
  openai_llm:
    _type: openai
    model_name: gpt-5-mini-2025-08-07
    api_type: responses

workflow:
  _type: responses_api_agent
  llm_name: openai_llm
  verbose: true
  handle_tool_errors: true

  # NAT tools are executed by the agent graph
  nat_tools: [current_datetime]

  # Built-in tools are bound to the LLM (for example, Code Interpreter)
  builtin_tools:
    - type: code_interpreter
      container:
        type: "auto"

  # Optional: Remote tools via Model Context Protocol
  mcp_tools:
    - type: mcp
      server_label: deepwiki
      server_url: https://mcp.deepwiki.com/mcp
      allowed_tools: [read_wiki_structure, read_wiki_contents]
      require_approval: never
```

## Configurable Options
The following are more ways you can configure your config file when using the Responses API and Agent:
- `llm_name`: The LLM to use. Must refer to an entry under `llms`.
- `verbose`: Defaults to `false`. When `true`, the agent logs input, output, and intermediate steps.
- `handle_tool_errors`: Defaults to `true`. When enabled, tool errors are returned to the model (instead of raising) so it can recover.
- `nat_tools`: A list of NeMo Agent toolkit tools (by function ref) that run in the agent graph.
- `builtin_tools`: A list of built-in tools to bind on the LLM. Availability depends on the selected model.
- `mcp_tools`: A list of MCP tool descriptors bound on the LLM, with fields `server_label`, `server_url`, `allowed_tools`, and `require_approval`.
- `max_iterations`: Defaults to `15`. Maximum number of tool invocations the agent may perform.
- `description`: Defaults to `Agent Workflow`. Used when the workflow is exported as a function.
- `parallel_tool_calls`: Defaults to `false`. If supported, allows the model runtime to schedule multiple tool calls in parallel.

## Running the Agent

Run from the repository root with a sample prompt:

```bash
nat run --config_file=examples/agents/tool_calling/configs/config-responses-api.yml --input "How many 0s are in the current time?"
```

## MCP Field Reference

When adding entries to `mcp_tools`, each object supports the following fields:

- `type`: Must be `mcp`.
- `server_label`: Short label for the server.
- `server_url`: URL of the MCP endpoint.
- `allowed_tools`: Optional allowlist of tool names the model may call.
- `require_approval`: One of `never`, `always`, or `auto`.
- `headers`: Optional map of HTTP headers to include when calling the server.
