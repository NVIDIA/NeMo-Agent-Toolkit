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

# NeMo Agent Toolkit Examples

Each NVIDIA NeMo Agent toolkit example demonstrates a particular feature or use case of the NeMo Agent toolkit library. Most of these contain a custom [workflow](../docs/source/tutorials/index.md) along with a set of custom tools ([functions](../docs/source/workflows/functions/index.md) in NeMo Agent toolkit). These examples can be used as a starting off point for creating your own custom workflows and tools. Each example contains a `README.md` file that explains the use case along with instructions on how to run the example.

## Examples Repository
In addition the examples in this repository, there are examples in the [NeMo-Agent-Toolkit-Examples](https://github.com/NVIDIA/NeMo-Agent-Toolkit-Examples) repository.

The difference between the examples in this repository and the NeMo-Agent-Toolkit-Examples repository is that the examples in this repository are maintained, tested, and updated with each release of the NeMo Agent toolkit. These examples have high quality standards and demonstrate a capability of the NeMo Agent toolkit.

The examples in the NeMo-Agent-Toolkit-Examples repository are community contributed and are tied to a specific version of the NeMo Agent toolkit, and do not need to demonstrate a specific capability of the library.


## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Example Categories](#example-categories)
  - [Getting Started](#getting-started)
  - [Agents](#agents)
  - [Advanced Agents](#advanced-agents)
  - [Control Flow](#control-flow)
  - [Custom Functions](#custom-functions)
  - [Evaluation and Profiling](#evaluation-and-profiling)
  - [Frameworks](#frameworks)
  - [Front Ends](#front-ends)
  - [Human In The Loop (HITL)](#human-in-the-loop-hitl)
  - [Memory](#memory)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
  - [Notebooks](#notebooks)
  - [Object Store](#object-store)
  - [Observability](#observability)
  - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [UI](#ui)
- [Documentation Guide Files](#documentation-guide-files)
  - [Locally Hosted LLMs](#locally-hosted-llms)
  - [Workflow Artifacts](#workflow-artifacts)

## Installation and Setup

To run the examples, install the NeMo Agent toolkit from source, if you haven't already done so, by following the instructions in [Install From Source](../docs/source/quick-start/installing.md#install-from-source).

## Example Categories

### Getting Started
- **[`scaffolding`](getting_started/scaffolding/README.md)**: Workflow scaffolding and project generation using automated commands and intelligent code generation
- **[`simple_web_query`](getting_started/simple_web_query/README.md)**: Basic LangSmith documentation agent that searches the internet to answer questions about LangSmith.
- **[`simple_calculator`](getting_started/simple_calculator/README.md)**: Mathematical agent with tools for arithmetic operations, time comparison, and complex calculations

### Agents
- **[`mixture_of_agents`](agents/mixture_of_agents/README.md)**: Multi-agent system with ReAct agent coordinating multiple specialized Tool Calling agents
- **[`react`](agents/react/README.md)**: ReAct (Reasoning and Acting) agent implementation for step-by-step problem-solving
- **[`rewoo`](agents/rewoo/README.md)**: ReWOO (Reasoning WithOut Observation) agent pattern for planning-based workflows
- **[`tool_calling`](agents/tool_calling/README.md)**: Tool-calling agent with direct function invocation capabilities

### Advanced Agents
- **[`AIQ Blueprint`](advanced_agents/aiq_blueprint/README.md)**: Blueprint documentation for the official NVIDIA AIQ Blueprint for building an AI agent designed for enterprise research use cases.
- **[`alert_triage_agent`](advanced_agents/alert_triage_agent/README.md)**: Production-ready intelligent alert triage system using LangGraph that automates system monitoring diagnostics with tools for hardware checks, network connectivity, performance analysis, and generates structured triage reports with root cause categorization
- **[`profiler_agent`](advanced_agents/profiler_agent/README.md)**: Performance profiling agent for analyzing NeMo Agent toolkit workflow performance and bottlenecks using Phoenix observability server with comprehensive metrics collection and analysis
- **[`vulnerability_analysis_blueprint`](advanced_agents/vulnerability_analysis_blueprint/README.md)**: Blueprint documentation for vulnerability analysis agents

### Control Flow
- **[`router_agent`](control_flow/router_agent/README.md)**: Configurable Router Agent that analyzes incoming requests and directly routes them to the most appropriate branch (other agents, functions or tools) based on request content
- **[`sequential_executor`](control_flow/sequential_executor/README.md)**: Linear tool execution pipeline that chains multiple functions together where each function's output becomes the input for the next function, with optional type compatibility checking and error handling

### Custom Functions
- **[`automated_description_generation`](custom_functions/automated_description_generation/README.md)**: Intelligent system that automatically generates descriptions for vector database collections by sampling and summarizing documents
- **[`plot_charts`](custom_functions/plot_charts/README.md)**: Multi-agent chart plotting system that routes requests to create different chart types (line, bar, etc.) from data

### Evaluation and Profiling
- **[`email_phishing_analyzer`](evaluation_and_profiling/email_phishing_analyzer/README.md)**: Evaluation and profiling configurations for the email phishing analyzer example
- **[`simple_calculator_eval`](evaluation_and_profiling/simple_calculator_eval/README.md)**: Evaluation and profiling configurations based on the basic simple calculator example
- **[`simple_web_query_eval`](evaluation_and_profiling/simple_web_query_eval/README.md)**: Evaluation and profiling configurations based on the basic simple web query example
- **[`swe_bench`](evaluation_and_profiling/swe_bench/README.md)**: Software engineering benchmark system for evaluating AI models on real-world coding tasks

### Frameworks
- **[`adk_demo`](frameworks/adk_demo/README.md)**: Minimal example using Google Agent Development Kit showcasing a simple weather time agent that can call tools (a function tool and an MCP tool)
- **[`agno_personal_finance`](frameworks/agno_personal_finance/README.md)**: Personal finance planning agent built with Agno framework that researches and creates tailored financial plans
- **[`haystack_deep_research_agent`](frameworks/haystack_deep_research_agent/README.md)**: Deep research agent using Haystack framework that combines web search and Retrieval Augmented Generation (RAG) capabilities with SerperDev API and OpenSearch
- **[`multi_frameworks`](frameworks/multi_frameworks/README.md)**: Supervisor agent coordinating LangChain/LangGraph, LlamaIndex, and Haystack agents for research, RAG, and chitchat tasks
- **[`semantic_kernel_demo`](frameworks/semantic_kernel_demo/README.md)**: Multi-agent travel planning system using Microsoft Semantic Kernel with specialized agents for itinerary creation, budget management, and report formatting, including long-term memory for user preferences

### Front Ends
- **[`simple_auth`](front_ends/simple_auth/README.md)**: Simple example demonstrating authentication and authorization using OAuth 2.0 Authorization Code Flow
- **[`simple_calculator_custom_routes`](front_ends/simple_calculator_custom_routes/README.md)**: Simple calculator example with custom API routing and endpoint configuration

### Human In The Loop (HITL)
- **[`por_to_jiratickets`](HITL/por_to_jiratickets/README.md)**: Project requirements to Jira ticket conversion with human oversight
- **[`simple_calculator_hitl`](HITL/simple_calculator_hitl/README.md)**: Human-in-the-loop version of the basic simple calculator that requests approval from the user before allowing the agent to make additional tool calls

### Memory
- **[`redis`](memory/redis/README.md)**: Basic long-term memory example using redis

### Model Context Protocol (MCP)
- **[`simple_auth_mcp`](MCP/simple_auth_mcp/README.md)**: Demonstrates how to use the NVIDIA NeMo Agent toolkit with MCP servers that require authentication using OAuth2 flows
- **[`simple_calculator_mcp`](MCP/simple_calculator_mcp/README.md)**: Demonstrates Model Context Protocol support using the basic simple calculator example

### Notebooks

**[Building an Agentic System](notebooks/README.md)**: Series of notebooks demonstrating how to build, connect, evaluate, profile and deploy an agentic system using the NeMo Agent toolkit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Agent-Toolkit/)

1. [Getting Started](notebooks/1_getting_started_with_nat.ipynb) - Getting started with the NeMo Agent toolkit
2. [Bringing Your Own Agent](notebooks/2_bringing_your_own_agent.ipynb) - Bringing your own agent to the NeMo Agent toolkit
3. [Adding Tools and Agents](notebooks/3_adding_tools_to_agents.ipynb) - Adding tools to your agentic workflow
4. [Multi-Agent Orchestration](notebooks/4_multi_agent_orchestration.ipynb) - Setting up a multi-agent orchestration workflow
5. [Observability, Evaluation, and Profiling](notebooks/5_observability_evaluation_and_profiling.ipynb) - Instrumenting with observability, evaluation and profiling tools

#### Brev Launchables

- **[`GPU Cluster Sizing`](notebooks/launchables/README.md)**: GPU Cluster Sizing with NeMo Agent Toolkit

### Object Store
- **[`user_report`](object_store/user_report/README.md)**: User report generation and storage system using object store (S3, MySQL, and/or memory)

### Observability
- **[`simple_calculator_observability`](observability/simple_calculator_observability/README.md)**: Basic simple calculator with integrated monitoring, telemetry, and observability features

### Retrieval Augmented Generation (RAG)
- **[`simple_rag`](RAG/simple_rag/README.md)**: Complete RAG system with Milvus vector database, document ingestion, and long-term memory using Mem0 platform

### UI
- **[`UI`](UI/README.md)**: Guide for integrating and using the web-based user interface of the NeMo Agent toolkit for interactive workflow management.

## Documentation Guide Files

### Locally Hosted LLMs
- **[`nim_config`](documentation_guides/locally_hosted_llms/nim_config.yml)**: Configuration for locally hosted NIM LLM models
- **[`vllm_config`](documentation_guides/locally_hosted_llms/vllm_config.yml)**: Configuration for locally hosted vLLM models

### Workflow Artifacts
- **`custom_workflow`**: Artifacts for the [Custom Workflow](../docs/source/tutorials/add-tools-to-a-workflow.md) tutorial
- **`text_file_ingest`**: Artifacts for the [Text File Ingest](../docs/source/tutorials/create-a-new-workflow.md) tutorial
