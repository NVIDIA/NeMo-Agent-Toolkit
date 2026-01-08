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
Refer to the License for the specific language governing permissions and
limitations under the License.
-->

# Observe Workflows

The NeMo Agent toolkit uses a flexible, plugin-based observability system that provides comprehensive support for configuring logging, tracing, and metrics for [workflows](../../build-workflows/about-building-workflows.md). Users can configure multiple telemetry exporters simultaneously from the available options or create custom integrations. The observability system:

- Uses an event-driven architecture with `IntermediateStepManager` publishing workflow events to a reactive stream
- Supports multiple concurrent telemetry exporters processing events asynchronously
- Provides built-in exporters for popular observability platforms (Phoenix, Langfuse, Weave, etc.)
- Enables custom telemetry exporter development for any observability service

These features enable developers to test their workflows locally and integrate observability seamlessly with their preferred monitoring stack.


### Compatibility with Previous Versions
As of v1.2, the span exporter exports attributes names prefixed with `nat` by default. In prior releases the attribute names were prefixed with `aiq`, to retain compatibility the `NAT_SPAN_PREFIX` environment variable can be set to `aiq`:
```bash
export NAT_SPAN_PREFIX=aiq
```

## Installation

The core observability features (console and file logging) are included by default. For advanced telemetry features like OpenTelemetry and Phoenix tracing, you need to install the optional telemetry extras.

If you have already installed the NeMo Agent toolkit from source, you can install package extras with the following commands:

```bash
# Install all optional telemetry extras
uv pip install -e '.[telemetry]'

# Install specific telemetry extras
uv pip install -e '.[data-flywheel]'
uv pip install -e '.[opentelemetry]'
uv pip install -e '.[phoenix]'
uv pip install -e '.[weave]'
uv pip install -e '.[ragaai]'
```

If you have not installed the NeMo Agent toolkit from source, you can install package extras with the following commands:

```bash
# Install all optional telemetry extras
uv pip install "nvidia-nat[telemetry]"

# Install specific telemetry extras
uv pip install "nvidia-nat[data-flywheel]"
uv pip install "nvidia-nat[opentelemetry]"
uv pip install "nvidia-nat[phoenix]"
uv pip install "nvidia-nat[weave]"
uv pip install "nvidia-nat[ragaai]"
```

## Available Tracing Exporters

The following table lists each exporter with its supported features and configuration guide:

| Provider | Integration Documentation | Supported Features |
| -------- | ------------------------- | ------------------ |
| [Catalyst](https://catalyst.raga.ai/) | [Observing with Catalyst](?provider=Catalyst#provider-integration-guides){.external} | Logging, Tracing |
| [NVIDIA Data Flywheel Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-data-flywheel) | [Observing with Data Flywheel](?provider=Data-Flywheel#provider-integration-guides){.external} | Logging, Tracing |
| [DBNL](https://distributional.com/) | [Observing with DBNL](?provider=DBNL#provider-integration-guides){.external} | Logging, Tracing |
| [Dynatrace](https://dynatrace.com/) | [Observing with Dynatrace](?provider=Dynatrace#provider-integration-guides){.external} | Logging, Tracing |
| [Galileo](https://galileo.ai/) | [Observing with Galileo](?provider=Galileo#provider-integration-guides){.external} | Logging, Tracing |
| [Langfuse](https://langfuse.com/) | Refer to the `examples/observability/simple_calculator_observability` example for usage details | Logging, Tracing |
| [LangSmith](https://www.langchain.com/langsmith) | Refer to the `examples/observability/simple_calculator_observability` example for usage details| Logging, Tracing |
| [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) | [Observing with OTel Collector](?provider=OTel-collector#provider-integration-guides){.external} | Logging, Tracing |
| [Patronus](https://www.patronus.ai/) | Refer to the `examples/observability/simple_calculator_observability` example for usage details | Logging, Tracing |
| [Phoenix](https://phoenix.arize.com/) | [Observing with Phoenix](?provider=Phoenix#provider-integration-guides){.external} | Logging, Tracing |
| [W&B Weave](https://wandb.ai/site/weave/) | [Observing with W&B Weave](?provider=Wandb-Weave#provider-integration-guides){.external} | Logging, Tracing, W&B Weave Redaction, Evaluation Metrics |

Additional options:

- **File Export** - Built-in file-based tracing for local development and debugging
- **Custom Exporters** - Refer to [Adding Telemetry Exporters](../../extend/custom-components/telemetry-exporters.md) for creating custom integrations

For complete configuration examples and setup instructions, check the `examples/observability/` directory.


## Configurable Components

The flexible observability system is configured using the `general.telemetry` section in the workflow configuration file. This section contains two subsections: `logging` and `tracing`, and each subsection can contain multiple telemetry exporters running simultaneously.

For a complete list of logging and tracing plugins and corresponding configuration settings use the following CLI commands.

```bash
# For all registered logging plugins
nat info components -t logging

# For all registered tracing plugins
nat info components -t tracing
```

Illustrated below is a sample configuration file demonstrating multiple exporters configured to run concurrently.

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: WARN
      file:
        _type: file
        path: ./.tmp/workflow.log
        level: DEBUG
    tracing:
      # Multiple exporters can run simultaneously
      phoenix:
        _type: phoenix
        # ... configuration fields
      weave:
        _type: weave
        # ... configuration fields
      file_backup:
        _type: file
        # ... configuration fields
```

### **Logging Configuration**

The `logging` section contains one or more logging providers. Each provider has a `_type` and optional configuration fields. The following logging providers are supported by default:

- `console`: Writes logs to the console.
- `file`: Writes logs to a file.

Available log levels:

- `DEBUG`: Detailed information for debugging.
- `INFO`: General information about the workflow.
- `WARNING`: Potential issues that should be addressed.
- `ERROR`: Issues that affect the workflow from running correctly.
- `CRITICAL`: Severe issues that prevent the workflow from continuing to run.

If a log level is specified, all logs at or above that level will be logged. For example, if the log level is set to `WARNING`, all logs at or above that level will be logged. If the log level is set to `ERROR`, all logs at or above that level will be logged.

### **Tracing Configuration**

The `tracing` section contains one or more tracing providers. Each provider has a `_type` and optional configuration fields. The observability system supports multiple concurrent exporters.

### NeMo Agent Toolkit Observability Components

The NeMo Agent toolkit observability system uses a generic, plugin-based architecture built on the Subject-Observer pattern. The system consists of several key components working together to provide comprehensive workflow monitoring:

#### Event Stream Architecture

- **`IntermediateStepManager`**: Publishes workflow events (`IntermediateStep` objects) to a reactive event stream, tracking function execution boundaries, [LLM](../../build-workflows/llms/index.md) calls, [tool](../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) usage, and intermediate operations.
- **Event Stream**: A reactive stream that broadcasts `IntermediateStep` events to all subscribed telemetry exporters, enabling real-time observability.
- **Asynchronous Processing**: All telemetry exporters process events asynchronously in background tasks, keeping observability "off the hot path" for optimal performance.

#### Telemetry Exporter Types

The system supports multiple exporter types, each optimized for different use cases:

- **Raw Exporters**: Process `IntermediateStep` events directly for simple logging, file output, or custom event processing.
- **Span Exporters**: Convert events into spans with lifecycle management, ideal for distributed tracing and span-based observability services.
- **OpenTelemetry Exporters**: Specialized exporters for OTLP-compatible services with pre-built integrations for popular observability platforms.
- **Advanced Custom Exporters**: Support complex business logic, stateful processing, and enterprise reliability patterns with circuit breakers and dead letter queues.

#### Processing Pipeline System

Each exporter can optionally include a processing pipeline that transforms, filters, batches, or aggregates data before export:

- **Processors**: Modular components for data transformation, filtering, batching, and format conversion.
- **Pipeline Composition**: Chain multiple processors together for complex data processing workflows.
- **Type Safety**: Generic type system ensures compile-time safety for data transformations through the pipeline.

#### Integration Components

- **{py:class}`nat.profiler.decorators`**: Decorators that wrap workflow and LLM framework context managers to inject usage-collection callbacks.
- **{py:class}`~nat.profiler.callbacks`**: Callback handlers that track usage statistics (tokens, time, inputs/outputs) and push them to the event stream. Supports LangChain/LangGraph, LLama Index, CrewAI, Semantic Kernel, and Google ADK frameworks.

### Registering a New Telemetry Provider as a Plugin

For complete information about developing and integrating custom telemetry exporters, including detailed examples, best practices, and advanced configuration options, Refer to [Adding Telemetry Exporters](../../extend/custom-components/telemetry-exporters.md).

## Provider Integration Guides

::::{tab-set}
  :sync-group: provider

  :::{tab-item} Catalyst
  :sync: Catalyst

    :::{include} ./observe-workflow-with-catalyst.md

  :::

  :::{tab-item} Data Flywheel
  :sync: Data-Flywheel

    :::{include} ./observe-workflow-with-data-flywheel.md

  :::

  :::{tab-item} DBNL
  :sync: DBNL

    :::{include} ./observe-workflow-with-dbnl.md

  :::

  :::{tab-item} Dynatrace
  :sync: Dynatrace

    :::{include} ./observe-workflow-with-dynatrace.md

  :::

  :::{tab-item} Galileo
  :sync: Galileo

    :::{include} ./observe-workflow-with-galileo.md

  :::

  :::{tab-item} OTel Collector
  :sync: OTel-collector

    :::{include} ./observe-workflow-with-otel-collector.md

  :::

  :::{tab-item} Phoenix
  :sync: Phoenix
   
    :::{include} ./observe-workflow-with-phoenix.md

  :::

  :::{tab-item} W&B Weave
  :sync: Wandb-Weave
  
    :::{include} ./observe-workflow-with-weave.md

  :::

::::

