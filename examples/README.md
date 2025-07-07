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

# Agent Intelligence Toolkit Examples

Each NVIDIA Agent Intelligence (AIQ) toolkit example demonstrates a particular feature or use case of the AIQ toolkit library. Most of these contain a custom [workflow](../docs/source/tutorials/index.md) along with a set of custom tools ([functions](../docs/source/workflows/functions/index.md) in AIQ toolkit). These examples can be used as a starting off point for creating your own custom workflows and tools. Each example contains a `README.md` file that explains the use case along with instructions on how to run the example.

## Example Categories

### Basic Examples ([`basic`](basic/))
- **[`scaffolding`](basic/scaffolding/)**: Workflow scaffolding and project generation using automated commands and intelligent code generation
- **[`functions`](basic/functions/)**: Function implementation examples including simple LangSmith documentation agent, calculator workflow, automated content generation, and chart visualization tools
- **[`frameworks`](basic/frameworks/)**: Integration examples with different AI frameworks including multi-framework patterns, Semantic Kernel integration, and personal finance applications

### Intermediate Examples ([`intermediate`](intermediate/))
- **[`evaluation_and_profiling`](intermediate/evaluation_and_profiling/)**: Performance evaluation and profiling tools for workflow optimization
- **[`UI`](intermediate/UI/)**: User interface examples for interactive AIQ toolkit applications
- **[`HITL`](intermediate/HITL/)**: Human-in-the-loop workflow examples for collaborative AI systems
- **[`MCP`](intermediate/MCP/)**: Model Context Protocol implementation examples
- **[`custom_routes`](intermediate/custom_routes/)**: Custom routing and API endpoint examples
- **[`observability`](intermediate/observability/)**: Monitoring and observability integration examples
- **[`RAG`](intermediate/RAG/)**: Retrieval-Augmented Generation examples with Milvus vector database and long-term memory
- **[`agents`](intermediate/agents/)**: Multi-agent system examples and agent coordination patterns

### Advanced Examples ([`advanced`](advanced/))
- **[`alert_triage_agent`](advanced/alert_triage_agent/)**: Complex alert triage system using LangGraph for automated system monitoring and diagnostics
- **[`profiler_agent`](advanced/profiler_agent/)**: Performance profiling agent for analyzing AIQ toolkit workflow performance using Phoenix server
- **[`AIQ-blueprint.md`](advanced/aiq-blueprint.md)**: Blueprint documentation for advanced system architectures

### Documentation Guides ([`documentation_guides`](documentation_guides/))
- **[`locally_hosted_llms`](documentation_guides/locally_hosted_llms/)**: Examples for hosting and using local LLM models with AIQ toolkit
- **[`workflows`](documentation_guides/workflows/)**: Workflow examples specifically created for documentation and tutorial purposes

To run the examples, install the AIQ toolkit from source, if you haven't already done so, by following the instructions in  [Install From Source](../docs/source/quick-start/installing.md#install-from-source) .
