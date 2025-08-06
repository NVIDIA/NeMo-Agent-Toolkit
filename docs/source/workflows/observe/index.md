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

# Observe Workflows

The NeMo Agent toolkit uses a flexible, plugin-based observability system that provides comprehensive support for configuring logging, tracing, and metrics for workflows. Users can configure multiple telemetry exporters simultaneously from the available options or create custom integrations. The observability system:

- Uses an event-driven architecture with `IntermediateStepManager` publishing workflow events to a reactive stream
- Supports multiple concurrent telemetry exporters processing events asynchronously
- Provides built-in exporters for popular observability platforms (Phoenix, Langfuse, Weave, etc.)
- Enables custom telemetry exporter development for any observability service

These features enable developers to test their workflows locally and integrate observability seamlessly with their preferred monitoring stack.

## Installation

The core observability features (console and file logging) are included by default. For advanced telemetry features like OpenTelemetry and Phoenix tracing, you need to install the optional telemetry extras:

```bash
# Install all optional telemetry extras
uv pip install -e '.[telemetry]'

# Install specific telemetry extras
uv pip install -e '.[opentelemetry]'
uv pip install -e '.[phoenix]'
uv pip install -e '.[weave]'
uv pip install -e '.[ragaai]'
```

## Configurable Components

The flexible observability system is configured using the `general.telemetry` section in the workflow configuration file. This section contains two subsections: `logging` and `tracing`, and each subsection can contain multiple telemetry exporters running simultaneously.

For a complete list of logging and tracing plugins and corresponding configuration settings use the following CLI commands.

```bash
# For all registered logging plugins
aiq info components -t logging

# For all registered tracing plugins
aiq info components -t tracing
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
        path: /tmp/workflow.log
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

### **Tracing Configuration**

The `tracing` section contains one or more tracing providers. Each provider has a `_type` and optional configuration fields. The observability system supports multiple concurrent exporters.

### Available Tracing Exporters

Each exporter has its own detailed configuration guide with complete setup instructions and examples:

- **[W&B Weave](https://wandb.ai/site/weave/)** - See [Observing with W&B Weave](./observe-workflow-with-weave.md)
- **[Phoenix](https://phoenix.arize.com/)** - See [Observing with Phoenix](./observe-workflow-with-phoenix.md)
- **[Galileo](https://galileo.ai/)** - See [Observing with Galileo](./observe-workflow-with-galileo.md)
- **[Langfuse](https://langfuse.com/)** - OTLP-compatible observability platform
- **[LangSmith](https://www.langchain.com/langsmith)** - LangChain's observability platform
- **[Patronus](https://patronus.ai/)** - AI evaluation and monitoring platform
- **[Catalyst](https://catalyst.raga.ai/)** - See [Observing with Catalyst](./observe-workflow-with-catalyst.md)
- **[OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)** - See [Observing with OTel Collector](./observe-workflow-with-otel-collector.md)
- **File Export** - Built-in file-based tracing for local development and debugging
- **Custom Exporters** - See [Adding Telemetry Exporters](../../extend/telemetry-exporters.md) for creating custom integrations

For complete configuration examples and setup instructions, refer to the individual guides linked above or check the `examples/observability/` directory.

### NeMo Agent Toolkit Observability Components

The NeMo Agent toolkit observability system uses a generic, plugin-based architecture built on the Subject-Observer pattern. The system consists of several key components working together to provide comprehensive workflow monitoring:

#### Event Stream Architecture

- **`IntermediateStepManager`**: Publishes workflow events (`IntermediateStep` objects) to a reactive event stream, tracking function execution boundaries, LLM calls, tool usage, and intermediate operations.
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

- **{py:class}`aiq.profiler.decorators`**: Decorators that wrap workflow and LLM framework context managers to inject usage-collection callbacks.
- **{py:class}`~aiq.profiler.callbacks`**: Callback handlers that track usage statistics (tokens, time, inputs/outputs) and push them to the event stream. Supports LangChain, LLama Index, CrewAI, and Semantic Kernel frameworks.

### Registering a New Telemetry Provider as a Plugin

For complete information about developing and integrating custom telemetry exporters, including detailed examples, best practices, and advanced configuration options, see [Adding Telemetry Exporters](../../extend/telemetry-exporters.md).

```{toctree}
:hidden:
:caption: Observe Workflows

Observing with Catalyst <./observe-workflow-with-catalyst.md>
Observing with Galileo <./observe-workflow-with-galileo.md>
Observing with OTEL Collector <./observe-workflow-with-otel-collector.md>
Observing with Phoenix <./observe-workflow-with-phoenix.md>
Observing with W&B Weave <./observe-workflow-with-weave.md>
```
