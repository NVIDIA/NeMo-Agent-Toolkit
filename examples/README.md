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

# NeMo Agent Toolkit Examples

Each NVIDIA NeMo Agent Toolkit example demonstrates a particular feature or use case of the NeMo Agent Toolkit library. Most of these contain a custom [workflow](../docs/source/get-started/tutorials/index.md) along with a set of custom tools ([functions](../docs/source/build-workflows/functions-and-function-groups/functions.md) in NeMo Agent Toolkit). These examples can be used as a starting off point for creating your own custom workflows and tools. Each example contains a `README.md` file that explains the use case along with instructions on how to run the example.

## Examples Repository
In addition the examples in this repository, there are examples in the [NeMo-Agent-Toolkit-Examples](https://github.com/NVIDIA/NeMo-Agent-Toolkit-Examples) repository.

The difference between the examples in this repository and the NeMo-Agent-Toolkit-Examples repository is that the examples in this repository are maintained, tested, and updated with each release of the NeMo Agent Toolkit. These examples have high quality standards and demonstrate a capability of the NeMo Agent Toolkit.

The examples in the NeMo-Agent-Toolkit-Examples repository are community contributed and are tied to a specific version of the NeMo Agent Toolkit, and do not need to demonstrate a specific capability of the library.


## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Notebooks](#notebooks)
- [Getting Started](#getting-started)
- **[NeMo Agent Toolkit Components](#nemo-agent-toolkit-components)**
  - [Agents](#agents)
  - [Advanced Agents](#advanced-agents)
  - [Configuration](#configuration)
  - [Control Flow](#control-flow)
  - [Custom Functions](#custom-functions)
  - [Frameworks](#frameworks)
  - [Front Ends](#front-ends)
  - [Memory](#memory)
  - [Object Store](#object-store)
  - [Human In The Loop (HITL)](#human-in-the-loop-hitl)
  - [UI](#ui)
- **[Connecting and Orchestrating Agents](#connecting-and-orchestrating-agents)**
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
  - [Agent2Agent Protocol (A2A)](#agent2agent-protocol-a2a)
- **[Observability, Evaluation, Profiling, and Finetuning](#observability-evaluation-profiling-and-finetuning)**
  - [Observability](#observability)
  - [Evaluation and Profiling](#evaluation-and-profiling)
  - [Finetuning](#finetuning)
- **[Platform Integrations](#platform-integrations)**
  - [Dynamo Integration](#dynamo-integration)
  - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Safety and Security](#safety-and-security)
- [Documentation Guide Files](#documentation-guide-files)
  - [Locally Hosted LLMs](#locally-hosted-llms)
  - [Workflow Artifacts](#workflow-artifacts)
- [Deploy Files](#deploy-files)

## Installation and Setup

To run the examples, install the NeMo Agent Toolkit from source, if you haven't already done so, by following the instructions in [Install From Source](../docs/source/get-started/installation.md#install-from-source).

## Notebooks

**[Building an Agentic System](notebooks/README.md)**: Series of notebooks demonstrating how to build, connect, evaluate, profile and deploy an agentic system using the NeMo Agent Toolkit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Agent-Toolkit/)

1. [Hello World](notebooks/hello_world.ipynb) - Installing NeMo Agent Toolkit and running a configuration-only workflow **[🟢 Beginner]**
2. [Getting Started](notebooks/getting_started_with_nat.ipynb) - Getting started with the NeMo Agent Toolkit **[🟢 Beginner]**
3. [Bringing Your Own Agent](notebooks/bringing_your_own_agent.ipynb) - Bringing your own agent to the NeMo Agent Toolkit **[🟢 Beginner]**
4. [Adding Tools and Agents](notebooks/adding_tools_to_agents.ipynb) - Adding tools to your agentic workflow **[🟢 Beginner]**
5. [MCP Client and Servers Setup](notebooks/mcp_setup_and_integration.ipynb) - Deploy and integrate MCP clients and servers with NeMo Agent Toolkit workflows **[🟢 Beginner]**
6. [Multi-Agent Orchestration](notebooks/multi_agent_orchestration.ipynb) - Setting up a multi-agent orchestration workflow **[🟨 Intermediate]**
7. [Observability, Evaluation, and Profiling](notebooks/observability_evaluation_and_profiling.ipynb) - Instrumenting with observability, evaluation and profiling tools **[🟨 Intermediate]**
8. [Optimizing Model Selection, Parameters, and Prompts](notebooks/optimize_model_selection.ipynb) - Use the NeMo Agent Toolkit Optimizer to compare models, parameters, and prompt variations **[🛑 Advanced]**

### Brev Launchables

- **[`GPU Cluster Sizing`](notebooks/launchables/README.md)**: GPU Cluster Sizing with NeMo Agent Toolkit **[🛑 Advanced]**

## Getting Started
- **[`scaffolding`](getting_started/scaffolding/README.md)**: Workflow scaffolding and project generation using automated commands and intelligent code generation **[🟢 Beginner]**
- **[`simple_web_query`](getting_started/simple_web_query/README.md)**: Basic LangSmith documentation agent that searches the internet to answer questions about LangSmith. **[🟢 Beginner]**
- **[`simple_calculator`](getting_started/simple_calculator/README.md)**: Mathematical agent with tools for arithmetic operations, time comparison, and complex calculations **[🟢 Beginner]**

## NeMo Agent Toolkit Components

### Agents
- **[`react`](agents/react/README.md)**: ReAct (Reasoning and Acting) agent implementation for step-by-step problem-solving **[🟢 Beginner]**
- **[`rewoo`](agents/rewoo/README.md)**: ReWOO (Reasoning WithOut Observation) agent pattern for planning-based workflows **[🟢 Beginner]**
- **[`tool_calling`](agents/tool_calling/README.md)**: Tool-calling agent with direct function invocation capabilities **[🟢 Beginner]**
- **[`auto_memory_wrapper`](agents/auto_memory_wrapper/README.md)**: Automatic memory wrapper agent that adds guaranteed memory capture and retrieval to any agent without requiring LLM memory tool invocation **[🟨 Intermediate]**
- **[`mixture_of_agents`](agents/mixture_of_agents/README.md)**: Multi-agent system with ReAct agent coordinating multiple specialized Tool Calling agents **[🟨 Intermediate]**

_Additional information can be found in the [Agents README](./agents/README.md)._

### Advanced Agents
- **[`AIQ Blueprint`](advanced_agents/aiq_blueprint/README.md)**: Blueprint documentation for the official NVIDIA AIQ Blueprint for building an AI agent designed for enterprise research use cases. **[🛑 Advanced]**
- **[`alert_triage_agent`](advanced_agents/alert_triage_agent/README.md)**: Production-ready intelligent alert triage system using LangGraph that automates system monitoring diagnostics with tools for hardware checks, network connectivity, performance analysis, and generates structured triage reports with root cause categorization **[🟨 Intermediate]**
- **[`profiler_agent`](advanced_agents/profiler_agent/README.md)**: Performance profiling agent for analyzing NeMo Agent Toolkit workflow performance and bottlenecks using Phoenix observability server with comprehensive metrics collection and analysis **[🟨 Intermediate]**
- **[`vulnerability_analysis_blueprint`](advanced_agents/vulnerability_analysis_blueprint/README.md)**: Blueprint documentation for vulnerability analysis agents **[🛑 Advanced]**

### Configuration
- **[`config_inheritance`](config_inheritance/README.md)**: Use YAML configuration inheritance in the NeMo Agent Toolkit to reduce duplication across similar configuration files **[🟢 Beginner]**

### Control Flow
- **[`router_agent`](control_flow/router_agent/README.md)**: Configurable Router Agent that analyzes incoming requests and directly routes them to the most appropriate branch (other agents, functions or tools) based on request content **[🟢 Beginner]**
- **[`sequential_executor`](control_flow/sequential_executor/README.md)**: Linear tool execution pipeline that chains multiple functions together where each function's output becomes the input for the next function, with optional type compatibility checking and error handling **[🟢 Beginner]**

### Custom Functions
- **[`automated_description_generation`](custom_functions/automated_description_generation/README.md)**: Intelligent system that automatically generates descriptions for vector database collections by sampling and summarizing documents **[🟢 Beginner]**
- **[`plot_charts`](custom_functions/plot_charts/README.md)**: Multi-agent chart plotting system that routes requests to create different chart types (line, bar, etc.) from data **[🟢 Beginner]**

### Frameworks
- **[`adk_demo`](frameworks/adk_demo/README.md)**: Minimal example using Google Agent Development Kit showcasing a simple weather time agent that can call tools (a function tool and an MCP tool) **[🟢 Beginner]**
- **[`agno_personal_finance`](frameworks/agno_personal_finance/README.md)**: Personal finance planning agent built with Agno framework that researches and creates tailored financial plans **[🟢 Beginner]**
- **[`autogen_demo`](frameworks/nat_autogen_demo/README.md)**: Minimal example using Microsoft AutoGen showcasing a traffic estimator agent that can call tools (a function tool and an MCP tool) and orchestrate multiple AutoGen agents **[🟨 Intermediate]**
- **[`haystack_deep_research_agent`](frameworks/haystack_deep_research_agent/README.md)**: Deep research agent using Haystack framework that combines web search and Retrieval Augmented Generation (RAG) capabilities with SerperDev API and OpenSearch **[🟨 Intermediate]**
- **[`langchain_deep_research`](frameworks/auto_wrapper/langchain_deep_research/README.md)**: An example that integrates any existing LangGraph agent with NeMo Agent Toolkit using the `langgraph_wrapper` workflow type **[🟨 Intermediate]**
- **[`multi_frameworks`](frameworks/multi_frameworks/README.md)**: Supervisor agent coordinating LangChain/LangGraph, LlamaIndex, and Haystack agents for research, RAG, and chitchat tasks **[🟨 Intermediate]**
- **[`semantic_kernel_demo`](frameworks/semantic_kernel_demo/README.md)**: Multi-agent travel planning system using Microsoft Semantic Kernel with specialized agents for itinerary creation, budget management, and report formatting, including long-term memory for user preferences **[🟢 Beginner]**
- **[`strands_demo`](frameworks/strands_demo/README.md)**: A minimal example showcasing a Strands agent that answers questions about Strands documentation using a curated URL knowledge base and the native Strands `http_request` tool **[🟨 Intermediate]**
- **[`strands_demo - bedrock_agentcore`](frameworks/strands_demo/bedrock_agentcore/README.md)**: Deploying NVIDIA NeMo Agent Toolkit with Strands on AWS AgentCore, including OpenTelemetry instrumentation for monitoring **[🛑 Advanced]**

### Front Ends
- **[`simple_auth`](front_ends/simple_auth/README.md)**: Simple example demonstrating authentication and authorization using OAuth 2.0 Authorization Code Flow **[🟨 Intermediate]**
- **[`simple_calculator_custom_routes`](front_ends/simple_calculator_custom_routes/README.md)**: Simple calculator example with custom API routing and endpoint configuration **[🟢 Beginner]**
- **[`per_user_workflow`](front_ends/per_user_workflow/README.md)**: Demonstrates the per-user workflow pattern in NeMo Agent Toolkit. With this pattern, each user gets their own isolated workflow and function instances with separate state. **[🟢 Beginner]**

### Memory
- **[`redis`](memory/redis/README.md)**: Basic long-term memory example using redis **[🟨 Intermediate]**

### Object Store
- **[`user_report`](object_store/user_report/README.md)**: User report generation and storage system using object store (S3, MySQL, and/or memory) **[🟨 Intermediate]**

### Human In The Loop (HITL)
- **[`por_to_jiratickets`](HITL/por_to_jiratickets/README.md)**: Project requirements to Jira ticket conversion with human oversight **[🟨 Intermediate]**
- **[`simple_calculator_hitl`](HITL/simple_calculator_hitl/README.md)**: Human-in-the-loop version of the basic simple calculator that requests approval from the user before allowing the agent to make additional tool calls **[🟢 Beginner]**

## UI
- **[`UI`](UI/README.md)**: Guide for integrating and using the web-based user interface of the NeMo Agent Toolkit for interactive workflow management. **[🛑 Advanced]**

## Connecting and Orchestrating Agents

### Model Context Protocol (MCP)
- **[`simple_calculator_mcp`](MCP/simple_calculator_mcp/README.md)**: Demonstrates an end-to-end MCP workflow with NVIDIA NeMo Agent Toolkit functioning as both MCP client and server. The MCP server is unprotected and intended for development and testing purposes **[🟢 Beginner]**
- **[`simple_calculator_mcp_protected`](MCP/simple_calculator_mcp_protected/README.md)**: Demonstrates an end-to-end OAuth2-protected MCP workflow with NVIDIA NeMo Agent Toolkit functioning as both MCP client and server. Demonstrates the use of per-user workflows to securely access the protected MCP server **[🟨 Intermediate]**
- **[`simple_auth_mcp`](MCP/simple_auth_mcp/README.md)**: Demonstrates a NVIDIA NeMo Agent Toolkit workflow connecting to a third-party MCP server that requires authentication using OAuth2 flows **[🟨 Intermediate]**
- **[`service_account_auth_mcp`](MCP/service_account_auth_mcp/README.md)**: Demonstrates how to use the NVIDIA NeMo Agent Toolkit with third-party MCP servers that support service account authentication **[🟨 Intermediate]**
- **[`kaggle_mcp`](MCP/kaggle_mcp/README.md)**: Demonstrates how to use the Kaggle MCP server with NVIDIA NeMo Agent Toolkit to interact with Kaggle's datasets, notebooks, models, and competitions **[🟨 Intermediate]**

### Agent2Agent Protocol (A2A)
- **[`currency_agent_a2a`](./A2A/currency_agent_a2a/README.md)**: Demonstrates a NVIDIA NeMo Agent Toolkit workflow connecting to a third-party A2A server, the LangGraph-based currency agent. The workflow acts as an A2A client to perform currency conversions and financial queries with time-based context **[🟢 Beginner]**
- **[`math_assistant_a2a`](./A2A/math_assistant_a2a/README.md)**: Demonstrates an end-to-end A2A workflow with NVIDIA NeMo Agent Toolkit functioning as both A2A client and server. The workflow performs mathematical calculations integrated with time queries and logical reasoning, combining remote calculator operations with local time services and conditional evaluation tools **[🟨 Intermediate]**
- **[`math_assistant_a2a_protected`](./A2A/math_assistant_a2a_protected/README.md)**: Demonstrates an end-to-end OAuth2-protected A2A workflow with NVIDIA NeMo Agent Toolkit functioning as both A2A client and server. The workflow performs mathematical calculations integrated with time queries and logical reasoning, with added OAuth2 authentication for secure per-user agent-to-agent communication **[🟨 Intermediate]**

## Observability, Evaluation, Profiling, and Finetuning

### Observability
- **[`simple_calculator_observability`](observability/simple_calculator_observability/README.md)**: Basic simple calculator with integrated monitoring, telemetry, and observability features **[🟨 Intermediate]**

### Evaluation and Profiling
- **[`email_phishing_analyzer`](evaluation_and_profiling/email_phishing_analyzer/README.md)**: Evaluation and profiling configurations for the email phishing analyzer example **[🟨 Intermediate]**
- **[`simple_calculator_eval`](evaluation_and_profiling/simple_calculator_eval/README.md)**: Evaluation and profiling configurations based on the basic simple calculator example **[🟨 Intermediate]**
- **[`simple_web_query_eval`](evaluation_and_profiling/simple_web_query_eval/README.md)**: Evaluation and profiling configurations based on the basic simple web query example **[🟨 Intermediate]**
- **[`swe_bench`](evaluation_and_profiling/swe_bench/README.md)**: Software engineering benchmark system for evaluating AI models on real-world coding tasks **[🛑 Advanced]**

### Finetuning
- **[`dpo_tic_tac_toe`](finetuning/dpo_tic_tac_toe/README.md)**: Demonstrates how to use the NeMo Agent Toolkit Test Time Compute (TTC) pipeline to generate preference data for Direct Preference Optimization (DPO) training, and submit training jobs to NVIDIA NeMo Customizer **[🛑 Advanced]**
- **[`rl_with_openpipe_art`](finetuning/rl_with_openpipe_art/README.md)**: Demonstrates how to use the NeMo Agent Toolkit finetuning harness with [OpenPipe ART](https://art.openpipe.ai/) (Agent Reinforcement Trainer) to improve an LLM's performance at playing Tic-Tac-Toe through reinforcement learning. **[🛑 Advanced]**

## Platform Integrations

### Dynamo Integration
- **[`react_benchmark_agent`](dynamo_integration/react_benchmark_agent/README.md)**: Walks through the complete process of running decision-only evaluations using the `react_benchmark_agent`: downloading data, configuring evaluations, running experiments, and analyzing results. **[🛑 Advanced]**
- **[`react_benchmark_agent - src - react_benchmark_agent`](dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/DEVELOPER_NOTES.md)**: Developer notes that map React Benchmark Agent configuration files to the underlying components, evaluators, and workflows. **[🛑 Advanced]**

_See the [Dynamo Integration README](dynamo_integration/README.md) for additional information_

### Retrieval Augmented Generation (RAG)
- **[`simple_rag`](RAG/simple_rag/README.md)**: Complete RAG system with Milvus vector database, document ingestion, and long-term memory using Mem0 platform **[🛑 Advanced]**

### Safety and Security
- **[`retail_agent`](safety_and_security/retail_agent/README.md)**: Outlines the features of the Safety and Security Engine (NASSE) included in NVIDIA NeMo Agent Toolkit and demonstrates its capabilities by assessing and improving the safety and security posture of an example Retail Agent **[🛑 Advanced]**

## Documentation Guide Files

_Additional information can be found in the Documentation Guides [README](./documentation_guides/README.md)._

### Locally Hosted LLMs
- **[`nim_config`](documentation_guides/locally_hosted_llms/nim_config.yml)**: Configuration for locally hosted NIM LLM models
- **[`vllm_config`](documentation_guides/locally_hosted_llms/vllm_config.yml)**: Configuration for locally hosted vLLM models

### Workflow Artifacts
- **`custom_workflow`**: Artifacts for the [Custom Workflow](../docs/source/get-started/tutorials/add-tools-to-a-workflow.md) tutorial
- **`text_file_ingest`**: Artifacts for the [Text File Ingest](../docs/source/get-started/tutorials/create-a-new-workflow.md) tutorial

## Deploy Files

The `deploy` directory contains files used by some examples for running services locally. Please consult the deploy [README](deploy/README.md) for more information.
