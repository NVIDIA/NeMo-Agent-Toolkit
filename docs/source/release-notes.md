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

# NVIDIA NeMo Agent Toolkit Release Notes
This section contains the release notes for [NeMo Agent toolkit](./index.md).

## Release 1.3.1
### Summary
This is a minor release with documentation updates, bug fixes, and non-breaking improvements.

### New Features
This release adds support for the Claude Sonnet 4.5 model and adds support for arbitrary JSON body types in custom routes.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/CHANGELOG.md) for a complete list of changes.

## Release 1.3.0
### Summary
This release introduces support for Google Agent Development Kit (ADK), adds control-flow agents, function groups, hyperparameter agent optimizer, LLM provider improvements, MCP server improvements, and Python 3.13 support. The toolkit continues to offer backwards compatibility, making the transition seamless for existing users.

### New Features
The following are the key features and improvements in this release:
* [ADK Support](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/reference/frameworks-overview.md): Supports Google Agent Development Kit (ADK). Adds tool calling, core observability, and LLM integration in this release.
* [Control-Flow Agents](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/about/index.md): [Sequential Executor](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/about/sequential-executor.md) (Linear Agent) and [Router Agent](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/about/router-agent.md) now control flow patterns of tool calls and sub-agents.
* [Function Groups](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/function-groups.md): Packages multiple related functions together so they share configuration, context, and resources.
* [Hyperparameter Agent Optimizer](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/reference/optimizer.md): Automates hyperparameter tuning and prompt engineering for workflows.
* [Introductory Notebook Improvements](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/examples/notebooks/README.md): Reorganizes getting started notebooks and adds Open in Colab links.
* [LLM Improvements](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/llms/index.md)
  - Adds LiteLLM Provider
  - Supports GPT-5 (`/chat/completions` endpoint only)
  - Adds Nemotron thinking configuration
* [MCP Improvements](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/docs/source/workflows/mcp/index.md)
  - Supports `streamable-http` - `sse` is no longer the default transport type.
  - Supports initial authorization - Enables connecting to MCP servers that require authentication.
  - Supports multiple MCP tools from a single configuration - Pulls in entire tool sets published by MCP servers or filters them based on user configuration.
  - Enhances CLI utilities for MCP servers and clients - Improves the `nat mcp` sub-command for querying, calling, and listing tools.
* Python 3.13 support

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.3/CHANGELOG.md) for a complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
