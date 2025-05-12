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

# AIQ Toolkit MCP Front-End

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. You can read more about MCP [here](https://modelcontextprotocol.io/introduction).

The MCP front-end in AIQ Toolkit allows you to expose your workflow functions as MCP-compatible tools that can be used by any MCP client. This enables seamless integration between AIQ Toolkit workflows and other MCP-compatible systems.

## Using the MCP Front-End

The MCP front-end is invoked using the `aiq mcp` command. This command starts an MCP server that exposes the functions from your workflow as MCP tools.

### Sample Usage

To start an MCP server exposing all tools from your workflow:

```bash
aiq mcp --config_file examples/simple_calculator/configs/config.yml
```

This will:
1. Load the workflow configuration from the specified file
2. Start an MCP server on the default host (localhost) and port (9901)
3. Expose all tools from the workflow as MCP tools

You can also specify a list of tool names to expose:

```bash
aiq mcp --config_file examples/simple_calculator/configs/config.yml \
  --tool_names calculator_multiply \
  --tool_names calculator_divide \
  --tool_names calculator_subtract \
  --tool_names calculator_inequality
```

### Listing MCP Tools

To list the tools exposed by the MCP server you can use the `aiq info mcp` command. This command acts as a MCP client and connects to the MCP server running on the specified URL (defaults to `http://localhost:9901/sse`).

```bash
aiq info mcp
```

Sample output:
```
calculator_multiply
calculator_inequality
calculator_divide
calculator_subtract
```

To get more information about a specific tool, use the `--detail` flag or the `--tool` flag followed by the tool name.

```bash
aiq info mcp --tool calculator_multiply
```

Sample output:
```
Tool: calculator_multiply
Description: This is a mathematical tool used to multiply two numbers together. It takes 2 numbers as an input and computes their numeric product as the output.
Input Schema:
{
  "properties": {
    "text": {
      "description": "",
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "CalculatorMultiplyInputSchema",
  "type": "object"
}
------------------------------------------------------------
```

### Accessing the MCP Tools

Once the MCP server is running, any MCP-compatible client can connect to it and use the exposed tools. For example, an MCP client could connect to `http://localhost:9901/sse` and use the `calculator_multiply` tool.

## Integration with MCP Clients

The AIQ Toolkit MCP front-end implements the Model Context Protocol specification, making it compatible with any MCP client. This allows for seamless integration with various systems that support MCP, including:

1. MCP-compatible LLM frameworks
2. Other agent frameworks that support MCP
3. Custom applications including AIQ Toolkit applications that implement the MCP client specification

### Sample Usage
To use the `math` tools exposed by the MCP server you can run the simple calculator example with the `config-mcp-math.yml` config file.
```bash
aiq run --config_file examples/simple_calculator/configs/config-mcp-math.yml --input "Is 2 times 2 greater than the current hour?"
```
With this configuration, the simple calculator workflow will act as a MCP client and connect to the MCP server running on the specified URL.
