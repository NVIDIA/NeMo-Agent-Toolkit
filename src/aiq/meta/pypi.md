<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http:/www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

![NVIDIA AgentIQ](https://media.githubusercontent.com/media/NVIDIA/AgentIQ/refs/heads/main/docs/source/_static/agentiq_banner.png "AgentIQ banner image")

# NVIDIA AgentIQ

AgentIQ is a flexible library designed to seamlessly integrate your enterprise agents—regardless of framework—with various data sources and tools. By treating agents, tools, and agentic workflows as simple function calls, AgentIQ enables true composability: build once and reuse anywhere.

## Key Features
AgentIQ works with any agentic framework, so you can use your current technology stack without replatforming.

### Workflows
![AgentIQ Workflow](https://media.githubusercontent.com/media/NVIDIA/AgentIQ/refs/heads/main/docs/source/_static/config_to_workflow.png "AgentIQ workflow image")

AgentIQ workflows are fully configurable via a YAML file and can be executed on any platform with a single command.

### Profiling
![AgentIQ Profiling](https://media.githubusercontent.com/media/NVIDIA/AgentIQ/refs/heads/main/docs/source/_static/profiler_token_scatter.png "AgentIQ profiling image")

Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.

### Observability
![AgentIQ Observability](https://media.githubusercontent.com/media/NVIDIA/AgentIQ/refs/heads/main/docs/source/_static/observability.png "AgentIQ observability image")

Monitor and debug your workflows with any OpenTelemetry-compatible observability tool. For example, visualize your workflows with Phoenix.

### Evaluation System
![AgentIQ Evaluation](https://media.githubusercontent.com/media/NVIDIA/AgentIQ/refs/heads/main/docs/source/_static/profiler_ragas_metrics.png  "AgentIQ evaluation image")

Validate and maintain accuracy of agentic workflows with built-in evaluation tools.

## Links
 * [Documentation](https://docs.nvidia.com/agentiq/latest/index.html): Explore the full documentation for AgentIQ.
 * [About AgentIQ](https://docs.nvidia.com/agentiq/latest/intro/why-agentiq.html): Learn more about the benefits of using AgentIQ.

## First time user?
 If this is your first time using AgentIQ, it is recommended to install the latest version from the [source repository](https://github.com/NVIDIA/AgentIQ?tab=readme-ov-file#get-started) on GitHub. This package is intended for users who are familiar with AgentIQ applications and need to add AgentIQ as a dependency to their project.

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AgentIQ/issues) if you have any feedback or feature requests.

## Acknowledgements

We would like to thank the following open source projects that made AgentIQ possible:

- [CrewAI](https://github.com/crewAIInc/crewAI)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Llama-Index](https://github.com/run-llama/llama_index)
- [Mem0ai](https://github.com/mem0ai/mem0)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [uv](https://github.com/astral-sh/uv)
