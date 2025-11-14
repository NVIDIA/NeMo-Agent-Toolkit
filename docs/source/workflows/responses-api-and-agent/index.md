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

# About the Responses API and Agent

The NVIDIA NeMo Agent toolkit supports OpenAI's Responses API through two complementary pieces:

- Configuring the LLM client mode using the `api_type` field 
- Integrating tool binding with the NeMo Agent toolkit dual-node graph using the dedicated workflow agent `_type: responses_api_agent`, designed for tool use with the Responses API.

The Responses API enables models to: 
- Use built-in tools such as Code Interpreter through `builtin_tools`.
- Connect to remote tools using Model Context Protocol (MCP) through `mcp_tools`, specifying fields such as `server_label` and `server_url`.
- Use toolkit tools through `nat_tools`, executed by the agent graph.

To configure your LLM agent for the Responses API and use the dedicated agent, refer to [Configure the Responses API and Agent](./responses-api-and-agent.md).