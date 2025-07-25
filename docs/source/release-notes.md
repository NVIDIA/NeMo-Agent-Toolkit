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

## Release 1.1.0
### Summary
* [Full Model Context Protocol (MCP) support](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/v1.1.0/docs/source/workflows/mcp/index.md). Workflows/tools can now be exposed as MCP servers.
* Deep integration with [Weights and Biases’ Weave](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/v1.1.0/docs/source/workflows/observe/observe-workflow-with-weave.md) for logging and tracing support.
* Addition of the [Agno](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/v1.1.0/examples/agno_personal_finance/README.md) LLM framework.
* A new [ReWOO agent](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/v1.1.0/examples/agents/rewoo/README.md) that improves on ReAct by removing the tool output from the LLM context, reducing token counts.
* A new [Alert Triage Agent example](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/v1.1.0/examples/alert_triage_agent/README.md) that demonstrates how to build a full application with NeMo Agent toolkit to automatically analyze system monitoring alerts, performs diagnostic checks using various tools, and generates structured triage reports with root cause categorization.
* Support for Python 3.11.
* Various other improvements.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/v1.1.0/CHANGELOG.md) for a complete list of changes.

## Release 1.0.0
### Summary
This is the first general release of NeMo Agent toolkit.

## LLM APIs
- NIM
- OpenAI

## Supported LLM Frameworks
- LangChain
- LlamaIndex

## Known Issues
- Faiss is currently broken on Arm64. This is a known issue [#72](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/72) caused by an upstream bug in the Faiss library [https://github.com/facebookresearch/faiss/issues/3936](https://github.com/facebookresearch/faiss/issues/3936).
- NeMo Agent toolkit applications must use the same name for both the distribution and root package. This is a current implementation limitation and will be addressed in a future release.
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
