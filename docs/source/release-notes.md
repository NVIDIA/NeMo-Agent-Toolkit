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

## Release 1.2.0
### Summary
The NeMo Agent toolkit, formerly known as Agent Intelligence (AIQ) toolkit, has been renamed in this release to align with the NVIDIA NeMo family of products. This release also brings significant new capabilities and improvements across authentication, resource management, observability, and developer experience. The toolkit continues to offer backwards compatibility, making the transition seamless for existing users.

The following are the key features and improvements in this release:
* [Authentication for Tool Calling](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/reference/api-authentication.md): Implement robust authentication mechanisms that enable secure and configurable access management for tool invocation within agent workflows.
* [Test Time Compute](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/reference/test-time-compute.md): Dynamically reallocate compute resources after model training, allowing agents to optimize reasoning, factual accuracy, and system robustness without retraining the base model.
* [Sizing Calculator](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/workflows/sizing-calc.md): Estimate GPU cluster requirements to support your target number of users and desired response times, simplifying deployment planning and scaling.
* [Object Store Integration](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/extend/object-store.md): Connect and manage data through supported object stores, improving agent extensibility and enabling advanced data workflows.
* [Enhanced Cursor Rules](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/tutorials/build-a-demo-agent-workflow-using-cursor-rules.md): Build new workflows or extend existing ones by leveraging cursor rules, making agent development faster and more flexible.
* [Interactive Notebooks](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/release/1.2/examples/notebooks): Access a suite of onboarding and example notebooks to accelerate agent workflow development, testing, and experimentation.
* [Observability Refactor](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/workflows/observe/index.md): Onboard new observability and monitoring platforms more easily, and take advantage of improved plug-in architecture for workflow inspection and analysis.
* [Examples Reorganization](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/examples/README.md): Organize examples by functionality, making it easier to find and use the examples.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/CHANGELOG.md) for a complete list of changes.

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
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
