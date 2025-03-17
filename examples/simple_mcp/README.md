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
# Simple Model Context Protocol (MCP) Tool Example

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. You can read more about MCP [here](https://modelcontextprotocol.io/introduction). AgentIQ implements an MCP Client Tool which allows AgentIQ workflows and functions to connect to and use tools served by remote MCP servers using server sent events.

## Usage
The MCP tool has a simple configuration consisting of three parameters:
```
url (str): The URL of the MCP server to connect to.
mcp_tool_name (str): The name of the tool to use. Since some servers may serve multiple tools, this maintains a one to
    one mapping of tools. The name needs to match the name of the tool provided by the server.
description (Optional[str]): Manually provided description. Most MCP tools provide a description, but if the provided
    description is insufficient (or nonexistent) a custom description can be provided.
```
The tool will produce a `pydantic` schema from the JSON schema provided by the MCP server, which can be used by the LLM
to aid tool calling. The tool can accept parameters as a JSON string or a python dictionary as long as they match the
generated `pydantic` schema.

## Example Workflow

This example workflow uses a locally hosted MCP SSE server running a [`fetch` tool](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch). This tool will fetch content from a URL (in Markdown by default) and provide it to the LLM as context.
By default, the workflow being run is the `react_agent`, which will use the `fetch` tool to pull data from the necessary URLs to answer the provided query.

### Installation and Setup
If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

To run this example do the following:
 1) Start up docker compose using the provided `docker-compose.yml` file.
 ```bash
 docker compose -f examples/simple_mcp/deployment/docker-compose.yml up -d
 ```
 The container will pull down the necessary code to run the server when it starts, so it may take a few minutes before the server is ready.
 You can inspect the logs by running
 ```bash
 docker compose -f examples/simple_mcp/deployment/docker-compose.yml logs
 ```
 The server is ready when you see the following:
 ```bash
 mcp-proxy-aiq  | INFO:     Started server process [1]
 mcp-proxy-aiq  | INFO:     Waiting for application startup.
 mcp-proxy-aiq  | INFO:     Application startup complete.
 mcp-proxy-aiq  | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
 ```

 2) In a new terminal, from the root of the AgentIQ repository run the workflow:
 ```bash
 source .venv/bin/activate
 aiq run --config_file=examples/simple_mcp/configs/config.yml --input="What is langchain?"
 ```

 The ReAct Agent will use the tool to answer the question
 ```console
 2025-03-11 16:13:29,922 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To answer this question, I need to find out what LangChain is. It's possible that it's a recent development or a concept that has been discussed online. I can use the internet to find the most up-to-date information about LangChain.

Action: mcp_url_tool
Action Input: {"url": "https://langchain.dev/", "max_length": 5000, "start_index": 0, "raw": false}


2025-03-11 16:13:29,924 - aiq.agent.react_agent.agent - INFO - Calling tool mcp_url_tool with input: {"url": "https://langchain.dev/", "max_length": 5000, "start_index": 0, "raw": false}
```
```console
Workflow Result:
["LangChain is a composable framework that supports developers in building, running, and managing applications powered by Large Language Models (LLMs). It offers a suite of products, including LangChain, LangGraph, and LangSmith, which provide tools for building context-aware and reasoning applications, deploying LLM applications at scale, and debugging, collaborating, testing, and monitoring LLM apps. LangChain's products are designed to help developers create reliable and efficient GenAI applications, and its platform is used by teams of all sizes across various industries."]
```
