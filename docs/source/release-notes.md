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

# NVIDIA NeMo Agent Toolkit Release Notes
This section contains the release notes for [NeMo Agent Toolkit](./index.md).

## Release 1.5.0
### Summary
This release expands runtime intelligence, framework-level performance acceleration, and production observability in the toolkit, while making workflow publishing to MCP ecosystems easier.

**Migration notice:** Release `1.5.0` includes packaging and compatibility refactors (including meta-package changes, eval package split, and import-path updates). Review the [Migration Guide](./resources/migration-guide.md#v150) before upgrading.

- [**Dynamo Runtime Intelligence:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/examples/dynamo_integration/latency_sensitivity_demo/README.md) Automatically infer per-request latency sensitivity from agent profiles and apply runtime hints for cache control, load-aware routing, and priority-aware serving.
- [**Agent Performance Primitives (APP):**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/packages/nvidia_nat_app/src/meta/pypi.md) Introduce framework-agnostic performance primitives that accelerate graph-based agent frameworks such as LangChain, CrewAI, and Agno with parallel execution, speculative branching, and node-level priority routing.
- [**LangSmith Native Integration:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/docs/source/run-workflows/observe/observe-workflow-with-langsmith.md) Observe end-to-end agent execution with native LangSmith tracing, run evaluation experiments, compare outcomes, and manage prompt versions across development and production workflows.
- [**FastMCP Workflow Publishing:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/docs/source/run-workflows/fastmcp-server.md) Publish NeMo Agent Toolkit workflows as MCP servers using the FastMCP runtime to simplify MCP-native deployment and integration.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/CHANGELOG.md) for the complete list of changes.

## Release 1.4.1
### Summary
This is a patch release with a critical bug fix for Python 3.11 users and documentation updates for `nat serve` CLI flags.

### 🐛 Bug Fixes
* fix(serve): ensure a single event loop for python 3.11 by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1558

### 📚 Documentation
* Document new `nat serve` CLI flags by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1562

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/CHANGELOG.md) for the complete list of changes.

## Release 1.4.0
### Summary
This release introduces initial support for several frameworks and integrations including A2A, AWS Strands, Amazon Bedrock AgentCore, Microsoft AutoGen, and NVIDIA Dynamo. In addition to new framework and integrations, an automatic agent wrapper for LangGraph Agents enables users to bring their own agent. Per-user functions enable deferred instantiation which provides per-user stateful functions, per-user resources, and other useful features. The toolkit continues to offer backwards compatibility, making the transition seamless for existing users.

- [**LangGraph Agent Automatic Wrapper:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/examples/frameworks/auto_wrapper/langchain_deep_research/README.md) Easily onboard existing LangGraph agents to NeMo Agent Toolkit. Use the automatic wrapper to access NeMo Agent Toolkit advanced features with very little modification of LangGraph agents.
- [**Automatic Reinforcement Learning (RL):**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/improve-workflows/finetuning/index.md) Improve your agent quality by fine-tuning open LLMs to better understand your agent's workflows, tools, and prompts. Perform GRPO with [OpenPipe ART](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/improve-workflows/finetuning/rl_with_openpipe.md) or DPO with [NeMo Customizer](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/improve-workflows/finetuning/dpo_with_nemo_customizer.md) using NeMo Agent Toolkit built-in evaluation system as a verifier.
- [**Initial NVIDIA Dynamo Integration:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/examples/dynamo_integration/README.md) Accelerate end-to-end deployment of agentic workflows with initial Dynamo support. Utilize the new agent-aware router to improve worker latency by predicting future agent behavior.
- [**A2A Support:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/components/integrations/a2a.md) Build teams of distributed agents using the A2A protocol.
- [**Safety and Security Engine:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/examples/safety_and_security/retail_agent/README.md) Strengthen safety and security workflows by simulating scenario-based attacks, profiling risk, running guardrail-ready evaluations, and applying defenses with red teaming. Validate defenses, profile risk, monitor behavior, and harden agents across any framework.
- [**Amazon Bedrock AgentCore and Strands Agents Support:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/components/integrations/frameworks.md#strands) Build agents using Strands Agents framework and deploy them securely on Amazon Bedrock AgentCore runtime.
- [**Microsoft AutoGen Support:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/components/integrations/frameworks.md#autogen) Build agents using the Microsoft AutoGen framework.
- [**Per-User Functions:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/extend/custom-components/custom-functions/per-user-functions.md) Use per-user functions for deferred instantiation, enabling per-user stateful functions, per-user resources, and other features.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/CHANGELOG.md) for a complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
