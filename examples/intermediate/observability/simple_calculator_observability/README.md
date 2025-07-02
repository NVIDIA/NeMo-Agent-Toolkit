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

# Simple Calculator - Observability and Tracing

This example demonstrates **observability and tracing capabilities** of the AIQ toolkit using the Simple Calculator workflow. Learn how to monitor, trace, and analyze your AI agent's behavior in real-time.

## 🎯 What You'll Learn

- **Distributed Tracing**: Track agent execution flow across components
- **Performance Monitoring**: Observe latency, token usage, and system metrics
- **Multi-Platform Integration**: Connect with popular observability tools
- **Real-time Analysis**: Monitor agent behavior during execution
- **Production Readiness**: Set up monitoring for deployed AI systems

## 🔗 Prerequisites

This example builds upon the [basic Simple Calculator](../../../basic/functions/simple_calculator/). Install it first:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

## 📦 Installation

```bash
uv pip install -e examples/intermediate/observability/simple_calculator_observability
```

## 🚀 Usage

### Phoenix Tracing (Local)
Start Phoenix and run with tracing enabled:

```bash
# Terminal 1: Start Phoenix
phoenix serve

# Terminal 2: Run with tracing
aiq run --config_file examples/intermediate/observability/simple_calculator_observability/configs/config-tracing.yml --input "What is 2 * 4?"
```

Visit `http://localhost:6006` to explore traces in Phoenix UI.

### LangFuse Integration
Configure LangFuse for production monitoring:

```bash
# Set your LangFuse credentials
export LANGFUSE_PUBLIC_KEY=<your_key>
export LANGFUSE_SECRET_KEY=<your_secret>
export LANGFUSE_HOST=<your_host>

aiq run --config_file examples/intermediate/observability/simple_calculator_observability/configs/config-langfuse.yml --input "Calculate 15 + 23"
```

### LangSmith Integration
Set up LangSmith for comprehensive monitoring:

```bash
export LANGCHAIN_API_KEY=<your_api_key>
export LANGCHAIN_PROJECT=<your_project>

aiq run --config_file examples/intermediate/observability/simple_calculator_observability/configs/config-langsmith.yml --input "Is 100 > 50?"
```

### Weave Integration
Use Weave for detailed workflow tracking:

```bash
export WANDB_API_KEY=<your_api_key>

aiq run --config_file examples/intermediate/observability/simple_calculator_observability/configs/config-weave.yml --input "What's the sum of 7 and 8?"
```

### Patronus Monitoring
Enable Patronus for AI safety and monitoring:

```bash
export PATRONUS_API_KEY=<your_api_key>

aiq run --config_file examples/intermediate/observability/simple_calculator_observability/configs/config-patronus.yml --input "Divide 144 by 12"
```

## 🔍 Key Features Demonstrated

- **Trace Visualization**: See complete execution paths
- **Performance Metrics**: Monitor response times and resource usage
- **Error Tracking**: Identify and diagnose issues quickly
- **Multi-Tool Support**: Choose the right observability platform
- **Production Monitoring**: Real-world deployment observability

## 📊 Available Configurations

| Config File | Platform | Purpose |
|-------------|----------|---------|
| `config-tracing.yml` | Phoenix | Local development tracing |
| `config-langfuse.yml` | LangFuse | Production monitoring |
| `config-langsmith.yml` | LangSmith | LangChain ecosystem monitoring |
| `config-weave.yml` | Weave | Workflow-focused tracking |
| `config-patronus.yml` | Patronus | AI safety and compliance |

## 🎛️ What Gets Traced

- **Agent Reasoning**: ReAct agent thought processes
- **Tool Calls**: Function invocations and responses
- **LLM Interactions**: Model calls, tokens, and latency
- **Error Events**: Failures and recovery attempts
- **Metadata**: Request context and custom attributes

This focused example shows how AIQ toolkit provides comprehensive observability essential for production AI applications.
