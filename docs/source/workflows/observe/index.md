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

The AIQ toolkit Observability Module provides support for configuring logging, tracing, and metrics for AIQ toolkit workflows. Users can configure telemetry options from a predefined list based on their preferences. The logging and tracing exporters:

- Listen for usage statistics pushed by `IntermediateStepManager`.
- Translate the usage statistics to OpenTelemetry format and push to the configured provider/method. (e.g., phoenix, OTelCollector, console, file)

These features enable AIQ toolkit developers to test their workflows locally and integrate observability seamlessly.

## Installation

The core observability features (console and file logging) are included by default. For advanced telemetry features like OpenTelemetry and Phoenix tracing, you need to install the optional telemetry dependencies:

```bash
uv pip install -e '.[telemetry]'
```

This will install:
- OpenTelemetry API and SDK for distributed tracing
- Arize Phoenix for visualization and analysis of LLM traces

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

To see the complete list of configuration fields for each provider, utilize the `aiq info -t logging` command which will display the configuration fields for each provider. For example:

```bash
$ aiq info -t logging
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
  - See [Observing with Phoenix](./observe-workflow-with-catalyst.md) for more information
- **Custom providers**
  - See [Registering a New Telemetry Provider as a Plugin](#registering-a-new-telemetry-provider-as-a-plugin) for more information


To see the complete list of configuration fields for each provider, utilize the `aiq info -t tracing` command which will display the configuration fields for each provider.


### AIQ Toolkit Observability Components

The Observability components `AsyncOtelSpanListener`, leverage the Subject-Observer pattern to subscribe to the `IntermediateStep` event stream pushed by `IntermediateStepManager`. Acting as an asynchronous event listener, `AsyncOtelSpanListener` listens for AIQ toolkit intermediate step events, collects and efficiently translates them into OpenTelemetry spans, enabling seamless tracing and monitoring.

- **Process events asynchronously** using a dedicated event loop.
- **Transform function execution boundaries** (`FUNCTION_START`, `FUNCTION_END`) and intermediate operations (`LLM_END`, `TOOL_END`) into OpenTelemetry spans.
- **Maintain function ancestry context** using `InvocationNode` objects, ensuring **distributed tracing across nested function calls**, while preserving execution hierarchy.
- **{py:class}`aiq.profiler.decorators`**: Defines decorators that can wrap each workflow or LLM framework context manager to inject usage-collection callbacks.
- **{py:class}`~aiq.profiler.callbacks`**: Directory that implements callback handlers. These handlers track usage statistics (tokens, time, inputs/outputs) and push them to the AIQ toolkit usage stats queue. AIQ toolkit profiling supports callback handlers for LangChain, LLama Index, CrewAI, and Semantic Kernel.


### Registering a New Telemetry Provider as a Plugin

AIQ toolkit allows users to register custom telemetry providers using the `@register_telemetry_exporter` decorator in {py:class}`aiq.observability.register`.

Example:
```bash
class PhoenixTelemetryExporter(TelemetryExporterBaseConfig, name="phoenix"):
    endpoint: str
    project: str


@register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder):

    from phoenix.otel import HTTPSpanExporter
    try:
        yield HTTPSpanExporter(endpoint=config.endpoint)
    except ConnectionError as ex:
        logger.warning("Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
                       ex,
                       exc_info=True)
    except Exception as ex:
        logger.error("Error in Phoenix telemetry Exporter\n %s", ex, exc_info=True)
```

```{toctree}
:hidden:
:caption: Observe Workflows

Observing with Catalyst <./observe-workflow-with-catalyst.md>
Observing with Phoenix <./observe-workflow-with-phoenix.md>
Observing with Galileo <./observe-workflow-with-galileo.md>
Observing with W&B Weave <./observe-workflow-with-weave.md>
```
