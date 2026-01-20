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


# Integrating Existing LangGraph Agents with NVIDIA NeMo Agent Toolkit

This example demonstrates how to integrate any existing LangGraph agent with NeMo Agent toolkit using the `langgraph_wrapper` workflow type.

We use LangGraph's **Deep Research agent** as a comprehensive example—a sophisticated multi-agent system for conducting web research with planning, sub-agent coordination, and synthesis. The integration techniques shown here apply to any LangGraph agent.

## What You'll Learn

The included Jupyter notebook (`langgraph_deep_research.ipynb`) provides a complete walkthrough:

1. Running an existing LangGraph agent through NeMo Agent toolkit without code changes
2. Making agents configurable with different components (LLMs, tools, embedders)
3. Adding Phoenix telemetry for observability
4. Evaluating agent performance with automated metrics

## Getting Started

### Prerequisites

Ensure NeMo Agent toolkit is installed. If not, follow the [Installation Guide](../../../../docs/source/get-started/installation.md).

### API Keys

- **NVIDIA Build API Key**: Required for section 3.0
- **Tavily API Key**: Required for web search functionality
- **Anthropic API Key** (optional): Required only for Section 2.0, which runs the original Deep Research agent with its default Claude model. You can skip Section 2.0 and start directly from Section 3.0 if you don't have an Anthropic API key.

### Launch the Notebook

From the **repository root**, run:

```bash
uv run jupyter notebook examples/frameworks/auto_wrapper/langchain_deep_research/langgraph_deep_research.ipynb
```

The notebook will guide you through:
- Setting up API keys (NVIDIA Build, Tavily)
- Installing dependencies automatically
- Running the agent with various configurations
- Adding telemetry and evaluation

All paths in the notebook are relative to the repository root, so make sure to launch from there.
