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

### Basic ([`basic`](basic/))
- **[`scaffolding`](basic/scaffolding/)**: Workflow scaffolding and project generation using automated commands and intelligent code generation
- **[`functions`](basic/functions/)**: Function implementation examples including:
  - [`simple`](basic/functions/simple/): LangSmith documentation agent that scrapes and answers questions about LangSmith using web retrieval and vector search
  - [`simple_calculator`](basic/functions/simple_calculator/): Mathematical agent with tools for arithmetic operations, time comparison, and complex calculations
  - [`automated_description_generation`](basic/functions/automated_description_generation/): Intelligent system that automatically generates descriptions for vector database collections by sampling and summarizing documents
  - [`plot_charts`](basic/functions/plot_charts/): Multi-agent chart plotting system that routes requests to create different chart types (line, bar, etc.) from data
- **[`frameworks`](basic/frameworks/)**: Integration examples with different AI frameworks including:
  - [`multi_frameworks`](basic/frameworks/multi_frameworks/): Supervisor agent coordinating LangChain, LlamaIndex, and Haystack agents for research, RAG, and chitchat tasks
  - [`agno_personal_finance`](basic/frameworks/agno_personal_finance/): Personal finance planning agent built with Agno framework that researches and creates tailored financial plans
  - [`semantic_kernel_demo`](basic/frameworks/semantic_kernel_demo/): Multi-agent travel planning system using Microsoft Semantic Kernel with specialized agents for itinerary creation, budget management, and report formatting, including long-term memory for user preferences

### Intermediate ([`intermediate`](intermediate/))
- **[`evaluation_and_profiling`](intermediate/evaluation_and_profiling/)**: Performance evaluation and profiling tools including:
  - [`swe_bench`](intermediate/evaluation_and_profiling/swe_bench/): Software engineering benchmark system for evaluating AI models on real-world coding tasks
  - [`simple_calculator_eval`](intermediate/evaluation_and_profiling/simple_calculator_eval/): Evaluation and profiling configurations based on the basic simple calculator example
  - [`simple_eval`](intermediate/evaluation_and_profiling/simple_eval/): Evaluation and profiling configurations based on the basic simple example
  - [`email_phishing_analyzer`](intermediate/evaluation_and_profiling/email_phishing_analyzer/): Security-focused email analysis system that detects phishing attempts using multiple LLMs, including its evaluation and profiling configurations
- **[`UI`](intermediate/UI/)**: User interface examples for interactive AIQ toolkit applications
- **[`HITL`](intermediate/HITL/)**: Human-in-the-loop workflow examples including:
  - [`simple_calculator_hitl`](intermediate/HITL/simple_calculator_hitl/): Human-in-the-loop version of the basic simple calculator that requests approval before increasing the agent's iteration limits
  - [`por_to_jiratickets`](intermediate/HITL/por_to_jiratickets/): Project requirements to Jira ticket conversion with human oversight
- **[`MCP`](intermediate/MCP/)**: Model Context Protocol implementation examples:
  - [`simple_calculator_mcp`](intermediate/MCP/simple_calculator_mcp/): Demonstrates Model Context Protocol support using the basic simple calculator example
- **[`custom_routes`](intermediate/custom_routes/)**: Custom routing and API endpoint examples:
  - [`simple_calculator_custom_routes`](intermediate/custom_routes/simple_calculator_custom_routes/): Basic simple calculator with custom API routing and endpoint configuration
- **[`observability`](intermediate/observability/)**: Monitoring and observability integration examples:
  - [`simple_calculator_observability`](intermediate/observability/simple_calculator_observability/): Basic simple calculator with integrated monitoring, telemetry, and observability features
- **[`RAG`](intermediate/RAG/)**: Retrieval-Augmented Generation examples:
  - [`simple_rag`](intermediate/RAG/simple_rag/): Complete RAG system with Milvus vector database, document ingestion, and long-term memory using Mem0 platform
- **[`agents`](intermediate/agents/)**: AI agent architecture examples showcasing 4 distinct agent patterns:
  - [`mixture_of_agents`](intermediate/agents/mixture_of_agents/): Multi-agent system with ReAct agent coordinating multiple specialized Tool Calling agents
  - [`react`](intermediate/agents/react/): ReAct (Reasoning and Acting) agent implementation for step-by-step problem solving
  - [`rewoo`](intermediate/agents/rewoo/): ReWOO (Reasoning WithOut Observation) agent pattern for planning-based workflows
  - [`tool_calling`](intermediate/agents/tool_calling/): Tool-calling agent with direct function invocation capabilities

### Advanced ([`advanced`](advanced/))
- **[`alert_triage_agent`](advanced/alert_triage_agent/)**: Production-ready intelligent alert triage system using LangGraph that automates system monitoring diagnostics with tools for hardware checks, network connectivity, performance analysis, and generates structured triage reports with root cause categorization
- **[`profiler_agent`](advanced/profiler_agent/)**: Performance profiling agent for analyzing AIQ toolkit workflow performance and bottlenecks using Phoenix observability server with comprehensive metrics collection and analysis
- **[`AIQ-blueprint.md`](advanced/aiq-blueprint.md)**: Blueprint documentation for advanced system architectures and design patterns

### Documentation Guides ([`documentation_guides`](documentation_guides/))
- **[`locally_hosted_llms`](documentation_guides/locally_hosted_llms/)**: Configuration examples for the basic simple LangSmith agent using locally hosted LLM models (NIM and vLLM configurations)
- **[`workflows`](documentation_guides/workflows/)**: Workflow examples for documentation and tutorials:
  - [`custom_workflow`](documentation_guides/workflows/custom_workflow/): Extended version of the basic simple example with multiple documentation sources (LangSmith and LangGraph)
  - [`text_file_ingest`](documentation_guides/workflows/text_file_ingest/): Text file processing and ingestion pipeline for document workflows

To run the examples, install the AIQ toolkit from source, if you haven't already done so, by following the instructions in  [Install From Source](../docs/source/quick-start/installing.md#install-from-source) .
