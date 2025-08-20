<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

![NVIDIA NeMo Agent Toolkit](https://media.githubusercontent.com/media/NVIDIA/NeMo-Agent-Toolkit/refs/heads/main/docs/source/_static/banner.png "NeMo Agent toolkit banner image")

# NVIDIA NeMo Agent Toolkit

NeMo Agent toolkit is a flexible library designed to seamlessly integrate your enterprise agents—regardless of framework—with various data sources and tools. By treating agents, tools, and agentic workflows as simple function calls, NeMo Agent toolkit enables true composability: build once and reuse anywhere.

## Key Features

- [**Framework Agnostic:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/extend/plugins.html) Works with any agentic framework, so you can use your current technology stack without replatforming.
- [**Reusability:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/extend/sharing-components.html) Every agent, tool, or workflow can be combined and repurposed, allowing developers to leverage existing work in new scenarios.
- [**Rapid Development:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/tutorials/index.html) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
- [**Profiling:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/workflows/profiler.html) Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.
- [**Observability:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/workflows/observe/observe-workflow-with-phoenix.html) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool, with examples using [Phoenix](https://docs.nvidia.com/nemo/agent-toolkit/1.3/workflows/observe/observe-workflow-with-phoenix.html) and [W&B Weave](https://docs.nvidia.com/nemo/agent-toolkit/1.3/workflows/observe/observe-workflow-with-weave.html).
- [**Evaluation System:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/workflows/evaluate.html) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [**User Interface:**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/quick-start/launching-ui.html) Use the NeMo Agent toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.
- [**MCP Compatibility**](https://docs.nvidia.com/nemo/agent-toolkit/1.3/workflows/mcp/mcp-client.html) Compatible with Model Context Protocol (MCP), allowing tools served by MCP Servers to be used as NeMo Agent toolkit functions.

With NeMo Agent toolkit, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

## Links
 * [Documentation](https://docs.nvidia.com/nemo/agent-toolkit/1.3/index.html): Explore the full documentation for NeMo Agent toolkit.

## First time user?
 If this is your first time using NeMo Agent toolkit, it is recommended to install the latest version from the [source repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit?tab=readme-ov-file#quick-start) on GitHub. This package is intended for users who are familiar with NeMo Agent toolkit applications and need to add NeMo Agent toolkit as a dependency to their project.

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) if you have any feedback or feature requests.

## Acknowledgements

We would like to thank the following open source projects that made NeMo Agent toolkit possible:

- [CrewAI](https://github.com/crewAIInc/crewAI)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Llama-Index](https://github.com/run-llama/llama_index)
- [Mem0ai](https://github.com/mem0ai/mem0)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [uv](https://github.com/astral-sh/uv)
