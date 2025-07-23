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

### Getting Started ([`getting_started`](getting_started/))
- **[`scaffolding`](getting_started/scaffolding/)**: Workflow scaffolding and project generation using automated commands and intelligent code generation
- **[`simple_web_query`](getting_started/simple_web_query/)**: Basic LangSmith documentation agent that searches the internet to answer questions about LangSmith.
- **[`simple_calculator`](getting_started/simple_calculator/)**: Mathematical agent with tools for arithmetic operations, time comparison, and complex calculations
- **[`simple_rag`](RAG/simple_rag/)**: Simple demonstration of using Retrieval Augmented Generation (RAG) with a vector database and document ingestion.

### Custom Functions ([`custom_functions`](custom_functions/))
- **[`automated_description_generation`](custom_functions/automated_description_generation/)**: Intelligent system that automatically generates descriptions for vector database collections by sampling and summarizing documents
- **[`plot_charts`](custom_functions/plot_charts/)**: Multi-agent chart plotting system that routes requests to create different chart types (line, bar, etc.) from data
- **[`simple_calculator`](getting_started/simple_calculator/)**: Mathematical agent with tools for arithmetic operations, time comparison, and complex calculations
- **[`por_to_jiratickets`](HITL/por_to_jiratickets/)**: Project requirements to Jira ticket conversion with human oversight

### Frameworks ([`frameworks`](frameworks/))
- **[`agno_personal_finance`](frameworks/agno_personal_finance/)**: Personal finance planning agent built with Agno framework that researches and creates tailored financial plans
- **[`multi_frameworks`](frameworks/multi_frameworks/)**: Supervisor agent coordinating LangChain, LlamaIndex, and Haystack agents for research, RAG, and chitchat tasks
- **[`semantic_kernel_demo`](frameworks/semantic_kernel_demo/)**: Multi-agent travel planning system using Microsoft Semantic Kernel with specialized agents for itinerary creation, budget management, and report formatting, including long-term memory for user preferences

### Agents ([`agents`](agents/))
- **[`react`](agents/react/)**: ReAct (Reasoning and Acting) agent implementation for step-by-step problem-solving
- **[`rewoo`](agents/rewoo/)**: ReWOO (Reasoning WithOut Observation) agent pattern for planning-based workflows
- **[`tool_calling`](agents/tool_calling/)**: Tool-calling agent with direct function invocation capabilities
- **[`mixture_of_agents`](agents/mixture_of_agents/)**: Multi-agent system with ReAct agent coordinating multiple specialized Tool Calling agents

### Front Ends ([`front_ends`](front_ends/))
- **[`simple_calculator_custom_routes`](front_ends/simple_calculator_custom_routes/)**: Simple calculator example with custom API routing and endpoint configuration

### Human In The Loop (HITL) ([`HITL`](HITL/))
- **[`por_to_jiratickets`](HITL/por_to_jiratickets/)**: Project requirements to Jira ticket conversion with human oversight
- **[`simple_calculator_hitl`](HITL/simple_calculator_hitl/)**: Human-in-the-loop version of the basic simple calculator that requests approval from the user before allowing the agent to make additional tool calls.

### MCP ([`MCP`](MCP/))
- **[`simple_calculator_mcp`](MCP/simple_calculator_mcp/)**: Demonstrates Model Context Protocol support using the basic simple calculator example

### Observability ([`observability`](observability/))
- **[`redact_pii`](observability/redact_pii/)**: Demonstrates how to use Weights & Biases (W&B) Weave with PII redaction
- **[`simple_calculator_observability`](observability/simple_calculator_observability/)**: Basic simple calculator with integrated monitoring, telemetry, and observability features
- **[`alert_triage_agent`](advanced_agents/alert_triage_agent/)**: Production-ready intelligent alert triage system using LangGraph that automates system monitoring diagnostics with tools for hardware checks, network connectivity, performance analysis, and generates structured triage reports with root cause categorization
- **[`profiler_agent`](advanced_agents/profiler_agent/)**: Performance profiling agent for analyzing AIQ toolkit workflow performance and bottlenecks using Phoenix observability server with comprehensive metrics collection and analysis
- **[`email_phishing_analyzer`](evaluation_and_profiling/email_phishing_analyzer/)**: Security-focused email analysis system that detects phishing attempts using multiple LLMs, including its evaluation and profiling configurations
- **[`text_file_ingest`](documentation_guides/workflows/text_file_ingest/)**: Text file processing and ingestion pipeline for document workflows

### Memory ([`memory`](memory/))
- **[`simple_rag`](RAG/simple_rag/)**: Complete RAG system with Milvus vector database, document ingestion, and long-term memory using Mem0 platform
- **[`semantic_kernel_demo`](frameworks/semantic_kernel_demo/)**: Multi-agent travel planning system using Microsoft Semantic Kernel with specialized agents for itinerary creation, budget management, and report formatting, including long-term memory for user preferences

### RAG ([`RAG`](RAG/))
- **[`simple_rag`](RAG/simple_rag/)**: Complete RAG system with Milvus vector database, document ingestion, and long-term memory using Mem0 platform

### Evaluation and Profiling ([`evaluation_and_profiling`](evaluation_and_profiling/))
- **[`swe_bench`](evaluation_and_profiling/swe_bench/)**: Software engineering benchmark system for evaluating AI models on real-world coding tasks
- **[`simple_calculator_eval`](evaluation_and_profiling/simple_calculator_eval/)**: Evaluation and profiling configurations based on the basic simple calculator example
- **[`simple_web_query_eval`](evaluation_and_profiling/simple_web_query_eval/)**: Evaluation and profiling configurations based on the basic simple web query example
- **[`alert_triage_agent`](advanced_agents/alert_triage_agent/)**: Evaluation and profiling configurations for the alert triage agent example
- **[`email_phishing_analyzer`](evaluation_and_profiling/email_phishing_analyzer/)**: Evaluation and profiling configurations for the email phishing analyzer example
- **[`text_file_ingest`](documentation_guides/workflows/text_file_ingest/)**: Evaluation and profiling configurations for the text file ingestion example

## Advanced Agents ([`advanced_agents`](advanced_agents/))
- **[`alert_triage_agent`](advanced_agents/alert_triage_agent/)**: Production-ready intelligent alert triage system using LangGraph that automates system monitoring diagnostics with tools for hardware checks, network connectivity, performance analysis, and generates structured triage reports with root cause categorization
- **[`profiler_agent`](advanced_agents/profiler_agent/)**: Performance profiling agent for analyzing AIQ toolkit workflow performance and bottlenecks using Phoenix observability server with comprehensive metrics collection and analysis
- **[`AIQ Blueprint`](advanced_agents/aiq_blueprint/)**: Blueprint documentation for the official NVIDIA AIQ Blueprint for building an AI agent designed for enterprise research use cases.

### Documentation Guides ([`documentation_guides`](documentation_guides/))
- **[`locally_hosted_llms`](documentation_guides/locally_hosted_llms/)**: Configuration examples for the basic simple LangSmith agent using locally hosted LLM models (NIM and vLLM configurations)
- **[`workflows`](documentation_guides/workflows/)**: Workflow examples for documentation and tutorials:
  - **[`custom_workflow`](documentation_guides/workflows/custom_workflow/)**: Extended version of the basic simple example with multiple documentation sources (LangSmith and LangGraph)
  - **[`text_file_ingest`](documentation_guides/workflows/text_file_ingest/)**: Text file processing and ingestion pipeline for document workflows

To run the examples, install the AIQ toolkit from source, if you haven't already done so, by following the instructions in  [Install From Source](../docs/source/quick-start/installing.md#install-from-source) .
