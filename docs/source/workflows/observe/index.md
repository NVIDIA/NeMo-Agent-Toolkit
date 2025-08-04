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

The NeMo Agent toolkit Observability Module provides support for configuring logging, tracing, and metrics for NeMo Agent toolkit workflows. Users can configure telemetry options from a predefined list based on their preferences. The logging and tracing exporters:

- Listen for usage statistics pushed by `IntermediateStepManager`.
- Translate the usage statistics to OpenTelemetry format and push to the configured provider/method. (e.g., phoenix, OTelCollector, console, file)

These features enable NeMo Agent toolkit developers to test their workflows locally and integrate observability seamlessly.

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

The observability module is configured using the `general.telemetry` section in the workflow configuration file. This section contains two subsections: `logging` and `tracing` and each subsection can contain one or more telemetry providers.

Illustrated below is a sample configuration file with all configurable components.

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: WARN
      file:
        _type: file
        path: /tmp/aiq_simple_calculator.log
        level: DEBUG
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: simple_calculator
```

### **Logging Configuration**

The `logging` section contains one or more logging providers. Each provider has a `_type` and optional configuration fields. The following logging providers are supported by default:

- `console`: Writes logs to the console.
- `file`: Writes logs to a file.

To see the complete list of configuration fields for each provider, utilize the `aiq info components -t logging` command which will display the configuration fields for each provider. For example:

```bash
$ aiq info components -t logging
                                                    AIQ Toolkit Search Results
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ package    ┃ version              ┃ component_type ┃ component_name ┃ description                                               ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ aiqtoolkit │ 1.2.0.dev15+g2322037 │ logging        │ console        │ A logger to write runtime logs to the console.            │
│            │                      │                │                │                                                           │
│            │                      │                │                │   Args:                                                   │
│            │                      │                │                │     _type (str): The type of the object.                  │
│            │                      │                │                │     level (str): The logging level of console logger.     │
├────────────┼──────────────────────┼────────────────┼────────────────┼───────────────────────────────────────────────────────────┤
│ aiqtoolkit │ 1.2.0.dev15+g2322037 │ logging        │ file           │ A logger to write runtime logs to a file.                 │
│            │                      │                │                │                                                           │
│            │                      │                │                │   Args:                                                   │
│            │                      │                │                │     _type (str): The type of the object.                  │
│            │                      │                │                │     path (str): The file path to save the logging output. │
│            │                      │                │                │     level (str): The logging level of file logger.        │
└────────────┴──────────────────────┴────────────────┴────────────────┴───────────────────────────────────────────────────────────┘
```

### **Tracing Configuration**

The `tracing` section contains one or more tracing providers. Each provider has a `_type` and optional configuration fields. The following tracing providers are supported by default:

- [**W&B Weave**](https://wandb.ai/site/weave/)
  - Example configuration:
    ```yaml
    tracing:
      weave:
        _type: weave
        project: "aiqtoolkit-demo"
    ```
  - See [Observing with W&B Weave](./observe-workflow-with-weave.md) for more information
- [**Phoenix**](https://phoenix.arize.com/)
  - Example configuration:
    ```yaml
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: "aiqtoolkit-demo"
    ```
  - See [Observing with Phoenix](./observe-workflow-with-phoenix.md) for more information
- [**Galileo**](https://galileo.ai/)
  - Example configuration:
    ```yaml
    tracing:
      galileo:
        _type: galileo
        endpoint: https://app.galileo.ai/api/galileo/otel/traces
        project: "aiqtoolkit-demo"
        logstream: "default"
        api_key: "<YOUR-GALILEO-API-KEY>"
    ```
  - See [Observing with Galileo](./observe-workflow-with-galileo.md) for more information
- [**Langfuse**](https://langfuse.com/)
  - Example configuration:
    ```yaml
    tracing:
      langfuse:
        _type: langfuse
        endpoint: http://localhost:3000/api/public/otel/v1/traces
    ```
- [**LangSmith**](https://www.langchain.com/langsmith)
  - Example configuration:
    ```yaml
    tracing:
      langsmith:
        _type: langsmith
        project: default
    ```
- [**Catalyst**](https://catalyst.raga.ai/)
  - Example configuration:
    ```yaml
    tracing:
      catalyst:
        _type: catalyst
        project: "aiqtoolkit-demo"
        dataset: "aiqtoolkit-dataset"
    ```
  - See [Observing with Catalyst](./observe-workflow-with-catalyst.md) for more information
- [**Generic OTel Collector**](./observe-workflow-with-otel-collector.md)
  - Example configuration:
  ```yaml
  tracing:
    otelcollector:
      _type: otelcollector
      project: "aiqtoolkit-demo"
      endpoint: "http://localhost:4318
  ```
  - See [Observing with OTel Collector](https://opentelemetry.io/docs/collector/) for more information
- **Custom providers**
  - See [Registering a New Telemetry Provider as a Plugin](#registering-a-new-telemetry-provider-as-a-plugin) for more information

To see the complete list of configuration fields for each provider, utilize the `aiq info components -t tracing` command which will display the configuration fields for each provider.

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
