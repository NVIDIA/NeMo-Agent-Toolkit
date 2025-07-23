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

# Simple Calculator with Observability and Tracing

This example demonstrates how to implement **observability and tracing capabilities** using the NVIDIA NeMo Agent toolkit. You'll learn to monitor, trace, and analyze your AI agent's behavior in real-time using the Simple Calculator workflow.

## Key Features

- **Multi-Platform Observability Integration:** Demonstrates integration with multiple observability platforms including Phoenix (local), Langfuse, LangSmith, Weave, Patronus, and RagAI Catalyst for comprehensive monitoring options.
- **Distributed Tracing Implementation:** Shows how to track agent execution flow across components with detailed trace visualization including agent reasoning, tool calls, and LLM interactions.
- **Performance Monitoring:** Demonstrates capturing latency metrics, token usage, resource consumption, and error tracking for production-ready AI system monitoring.
- **Development and Production Patterns:** Provides examples for both local development tracing (Phoenix) and production monitoring setups with various enterprise observability platforms.
- **Comprehensive Telemetry Collection:** Shows automatic capture of agent thought processes, function invocations, model calls, error events, and custom metadata for complete workflow visibility.

## What You'll Learn

- **Distributed tracing**: Track agent execution flow across components
- **Performance monitoring**: Observe latency, token usage, and system metrics
- **Multi-platform integration**: Connect with popular observability tools
- **Real-time analysis**: Monitor agent behavior during execution
- **Production readiness**: Set up monitoring for deployed AI systems

## Prerequisites

Before starting this example, you need:

1. **Agent toolkit**: Ensure you have the Agent toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.
2. **Base workflow**: This example builds upon the Getting Started [Simple Calculator](../../getting_started/simple_calculator/) example. Make sure you are familiar with the example before proceeding.
3. **Observability platform**: Access to at least one of the supported platforms (Phoenix, Langfuse, LangSmith, Weave, or Patronus)

## Installation

Install this observability example:

```bash
uv pip install -e examples/observability/simple_calculator_observability
```

## Getting Started

### Phoenix Tracing (Local Development)

Phoenix provides local tracing capabilities perfect for development and testing.

1. Start Phoenix in a separate terminal:

```bash
phoenix serve
```

2. Run the workflow with tracing enabled:

```bash
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-tracing.yml --input "What is 2 * 4?"
```

3. Open your browser to `http://localhost:6006` to explore traces in the Phoenix UI.

### Production Monitoring Platforms

For production deployments, you can integrate with these observability platforms:

#### Langfuse Integration

Langfuse provides production-ready monitoring and analytics.

1. Set your Langfuse credentials:

```bash
export LANGFUSE_PUBLIC_KEY=<your_key>
export LANGFUSE_SECRET_KEY=<your_secret>
export LANGFUSE_HOST=<your_host>
```

2. Run the workflow:

```bash
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-langfuse.yml --input "Calculate 15 + 23"
```

#### LangSmith Integration

LangSmith offers comprehensive monitoring within the LangChain ecosystem.

1. Set your LangSmith credentials:

```bash
export LANGCHAIN_API_KEY=<your_api_key>
export LANGCHAIN_PROJECT=<your_project>
```

2. Run the workflow:

```bash
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-langsmith.yml --input "Is 100 > 50?"
```

#### Weave Integration

Weave provides detailed workflow tracking and visualization.

1. Set your Weights & Biases API key:

```bash
export WANDB_API_KEY=<your_api_key>
```

2. Run the workflow:

```bash
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-weave.yml --input "What's the sum of 7 and 8?"
```

For detailed Weave setup instructions, see the [Fine-grained Tracing with Weave](../../../docs/source/workflows/observe/observe-workflow-with-weave.md) guide.

#### AI Safety Monitoring with Patronus

Patronus enables AI safety monitoring and compliance tracking.

1. Set your Patronus API key:

```bash
export PATRONUS_API_KEY=<your_api_key>
```

2. Run the workflow:

```bash
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-patronus.yml --input "Divide 144 by 12"
```

#### RagAI Catalyst Integration

Transmit traces to RagAI Catalyst.

1. Set your Patronus API key:

```bash
export CATALYST_ACCESS_KEY=<your_api_key>
export CATALYST_SECRET_KEY=<your_secret_key>
export CATALYST_ENDPOINT=<your_endpoint>
```

2. Run the workflow:

```bash
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-catalyst.yml --input "Divide 144 by 12"
```


## Configuration Files

The example includes multiple configuration files for different observability platforms:

| Configuration File | Platform | Best For |
|-------------------|----------|----------|
| `config-tracing.yml` | Phoenix | Local development and testing |
| `config-langfuse.yml` | Langfuse | Production monitoring and analytics |
| `config-langsmith.yml` | LangSmith | LangChain ecosystem integration |
| `config-weave.yml` | Weave | Workflow-focused tracking |
| `config-patronus.yml` | Patronus | AI safety and compliance monitoring |
| `config-catalyst.yml` | Catalyst | RagaAI Catalyst integration |

## What Gets Traced

The Agent toolkit captures comprehensive telemetry data including:

- **Agent reasoning**: ReAct agent thought processes and decision-making
- **Tool calls**: Function invocations, parameters, and responses
- **LLM interactions**: Model calls, token usage, and latency metrics
- **Error events**: Failures, exceptions, and recovery attempts
- **Custom metadata**: Request context, user information, and custom attributes

## Key Features Demonstrated

- **Trace visualization**: Complete execution paths and call hierarchies
- **Performance metrics**: Response times, token usage, and resource consumption
- **Error tracking**: Automated error detection and diagnostic information
- **Multi-platform support**: Flexibility to choose the right observability tool
- **Production monitoring**: Real-world deployment observability patterns
