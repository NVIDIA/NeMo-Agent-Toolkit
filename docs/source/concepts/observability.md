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

# NVIDIA AgentIQ Observability

The AgentIQ Observability Module provides support for configurable telemetry setup to do logging tracing and metrics for AgentIQ workflows.
- Enables users to configure telemetry options from a predefined list based on their preferences.
- Listens real-time usage statistics pushed by `IntermediateStepManager`.
- Translates the usage statistics to OpenTelemetry format and push to the configured provider/method. (e.g., phoenix, OTelCollector, console, file)

These features enable AgentIQ developers to test their workflows locally and integrate observability seamlessly.

## Configurable Components

Users can set up telemetry configuration within the workflow configuration file.

### **Logging Configuration**
Users can write logs to:
- **Console** (`console`)
- **Temporary file** (`file`)
- **Both** (by specifying both options)

#### **Configuration Fields**
- **`_type`**: Accepted values → `console`, `file`
- **`level`**: Log level (e.g., `DEBUG`, `INFO`, `WARN`, `ERROR`)
- **`path`** *(for file logging only)*: File path where logs will be stored.

### **Tracing Configuration**
Users can set up tracing using:
- **Phoenix**
- **Custom providers** *(See registration section below.)*

#### **Configuration Fields**
- **`_type`**: The name of the registered provider.
- **`endpoint`**: The provider’s listening endpoint.
- **`project`**: The associated project name.


Sample Configuration:
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


### AgentIQ Observability Components

The Observability components `AsyncOtelSpanListener`, leverage the Subject-Observer pattern to subscribe to the `IntermediateStep` event stream pushed by `IntermediateStepManager`. Acting as an asynchronous event listener, `AsyncOtelSpanListener` listens for AgentIQ intermediate step events, collects and efficiently translates them into OpenTelemetry spans, enabling seamless tracing and monitoring.

- **Process events asynchronously** using a dedicated event loop.
- **Transform function execution boundaries** (`FUNCTION_START`, `FUNCTION_END`) and intermediate operations (`LLM_END`, `TOOL_END`) into OpenTelemetry spans.
- **Maintain function ancestry context** using `InvocationNode` objects, ensuring **distributed tracing across nested function calls**, while preserving execution hierarchy.
- **{py:class}`aiq.profiler.decorators`**: Defines decorators that can wrap each workflow or LLM framework context manager to inject usage-collection callbacks.
- **{py:class}`~aiq.profiler.callbacks`**: Directory that implements callback handlers. These handlers track usage statistics (tokens, time, inputs/outputs) and push them to the AgentIQ usage stats queue. AgentIQ profiling supports callback handlers for LangChain, LLama Index, CrewAI, and Semantic Kernel.


### Registering a New Telemetry Provider as a Plugin

AgentIQ allows users to register custom telemetry providers using the `@register_telemetry_exporter` decorator in {py:class}`aiq.observability.register`.

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
