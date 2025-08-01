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

# Adding Telemetry Exporters to NVIDIA NeMo Agent Toolkit

> **Note**: The code examples in this guide are pseudo code designed to illustrate the programming interface and key concepts. They focus on demonstrating the structure and flow rather than providing complete, runnable implementations. Use these examples to understand the interface patterns and adapt them to your specific use case.

Telemetry exporters are plugins that send telemetry data (traces, spans, and intermediate steps) from NeMo Agent toolkit workflows to external observability services. This guide provides a comprehensive overview of how to create and register custom telemetry exporters.

## Why Use Telemetry Exporters?

Telemetry exporters solve critical observability challenges in Agentic AI workflows:

### **Production Monitoring**
- **Track workflow performance**: Monitor execution times, success rates, and resource usage across your AI agents
- **Identify bottlenecks**: Discover slow LLM calls, inefficient tool usage, or processing delays
- **Real-time alerting**: Get notified when workflows fail or performance degrades

### **Debugging and Troubleshooting**
- **Trace execution flow**: Follow the complete path of requests through your agent workflows
- **Debug failures**: Understand exactly where and why workflows fail with detailed error context
- **Inspect intermediate data**: See inputs, outputs, and transformations at each step

### **Analytics and Insights**
- **Usage patterns**: Understand how users interact with your AI agents
- **Cost optimization**: Track token usage, API calls, and resource consumption
- **Performance analysis**: Identify trends and optimization opportunities

### **Integration and Compliance**
- **Enterprise observability**: Connect to existing monitoring infrastructure (Datadog, etc.)
- **Compliance requirements**: Maintain audit trails and detailed logs for regulatory compliance
- **Custom dashboards**: Build specialized visualizations for your specific use cases

### **Common Use Cases**

| Scenario | Benefit | Recommended Exporter |
|----------|---------|---------------------|
| **Development debugging** | Quick local inspection of workflow behavior | RawExporter |
| **Production monitoring** | Real-time performance tracking and alerting using a span-based data structure | SpanExporter |
| **Enterprise integration** | Connect to existing OpenTelemetry based observability stack | OtelSpanExporter|
| **Custom analytics** | Specialized data processing and visualization | ProcessingExporter |
| **Compliance auditing** | Detailed audit trails and data retention | FileExporter |

**Without telemetry exporters**, you're operating blind - unable to understand performance, debug issues, or optimize your AI workflows. **With telemetry exporters**, you gain complete visibility into your agent operations, enabling confident production deployment and continuous improvement.

## Existing Telemetry Exporters

To view the list of locally installed and registered telemetry exporters, run the following command:

```bash
aiq info components -t tracing
```

Examples of existing telemetry exporters include:

- **Phoenix**: Exports traces to Arize Phoenix for visualization
- **Weave**: Exports traces to Weights & Biases Weave
- **OtelCollector**: Exports traces to OpenTelemetry-compatible services
- **Catalyst**: Exports traces to RagaAI Catalyst
- **File**: Exports traces to local files

## Quick Start: Your First Telemetry Exporter

Want to get started quickly? Here's a minimal working example that creates a console exporter to print traces to the terminal:

```python
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.observability.exporter.raw_exporter import RawExporter
from aiq.data_models.intermediate_step import IntermediateStep

# Step 1: Define configuration
class ConsoleTelemetryExporter(TelemetryExporterBaseConfig, name="console"):
    prefix: str = Field(default="[TRACE]", description="Prefix for console output")

# Step 2: Create exporter class
class ConsoleExporter(RawExporter[IntermediateStep, IntermediateStep]):
    """
    RawExporter[IntermediateStep, IntermediateStep] means:
    - Input: IntermediateStep (raw workflow events)
    - Output: IntermediateStep (no transformation needed)
    """
    def __init__(self, prefix: str = "[TRACE]", context_state=None):
        super().__init__(context_state=context_state)
        self.prefix = prefix

    async def export_processed(self, item: IntermediateStep):
        print(f"{self.prefix} {item.event_type}: {item.name}")
        # IntermediateStep contains workflow events with fields like:
        # - event_type: The type of event (e.g., "function_call", "llm_response")
        # - name: The name of the step or component
        # - metadata: Additional context and data

# Step 3: Register the exporter
@register_telemetry_exporter(config_type=ConsoleTelemetryExporter)
async def console_telemetry_exporter(config: ConsoleTelemetryExporter, builder: Builder):
    yield ConsoleExporter(prefix=config.prefix)
```

**Usage in workflow.yaml:**

```yaml
general:
  telemetry:
    tracing:
      console_exporter:
        type: console
        prefix: "[MY_APP]"
```

That's it! Your exporter will now print trace information to the console. Let's explore more advanced features below.

## Key Concepts

Before diving into advanced features, here are the core concepts:

1. **Configuration Class**: Defines the settings your exporter needs (endpoints, API keys, etc.) and its registered name
2. **Exporter Class**: Contains the logic to process and export trace data
3. **Registration Function**: Connects your configuration to your exporter implementation
4. **Processing Pipeline**: Optional transformations applied to data before export
5. **Isolation**: Ensures concurrent workflows don't interfere with each other

**The Three-Step Pattern:**

1. Define what settings you need (configuration)
2. Implement how to export data (exporter class)
3. Register the exporter with the toolkit (registration function)

## Understanding Telemetry Exporters

Telemetry exporters in NeMo Agent toolkit are responsible for:

1. **Event Subscription**: Listening to workflow intermediate steps
2. **Data Processing**: Transforming raw events into the target format
3. **Export**: Sending processed data to target destinations
4. **Lifecycle Management**: Handling startup, shutdown, and error conditions

### Telemetry Data Flow

The telemetry export system routes workflow events through different exporter types to various destinations:

```{mermaid}
graph TD
    A[Workflow Events] --> B[Event Stream]
    B --> C[Telemetry Exporter]
    C --> D[Processing Pipeline]
    D --> E[Raw Exporter]
    D --> F[Span Exporter]
    D --> G[OpenTelemetry Exporter]
    E --> H[File/Console Output]
    F --> I[Custom Service]
    G --> J[OTLP Compatible Service]

    style A fill:#e1f5fe
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
```

### Exporter Types

NeMo Agent toolkit supports several types of exporters based on the data they handle:

```{mermaid}
graph LR
    A["IntermediateStep"] --> B["Raw Exporter"]
    A --> C["Span Exporter"]
    A --> D["OpenTelemetry Exporter"]

    B --> E["Direct Processing<br/>File, Console, Custom"]
    C --> F["Span Processing<br/>Weave, HTTP APIs, Databases"]
    D --> G["OTLP Processing<br/>Datadog, Phoenix, Otel Collectors"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style G fill:#e8f5e8
```

#### Choosing the Right Exporter Type

The following table helps you choose the appropriate exporter type for your use case:

| Exporter Type | Use When | Best For | Complexity | Development Time |
|---------------|----------|----------|------------|------------------|
| **Raw Exporter** | Simple file/console output<br/>Basic event processing<br/>Development and debugging | Local development<br/>File-based logging<br/>Custom data formats | Low | 30 minutes |
| **Span Exporter** | HTTP API integration<br/>Custom observability services<br/>Non-OTLP backends | Production HTTP APIs<br/>Databases<br/>Custom dashboards | Medium | 2-4 hours |
| **OpenTelemetry Exporter** | OTLP-compatible services<br/>Standard observability tools<br/>Enterprise monitoring | Jaeger, Tempo<br/>Observability platforms<br/>Standard compliance | Low | 15-30 minutes |
| **Advanced Custom Exporter** | Complex business logic<br/>Stateful data processing<br/>Multi-system integrations | Enterprise reliability patterns<br/>Custom analytics platforms<br/>High-volume production workloads | High | 1-2 days |

**Quick Decision Guide:**
- **Using standard observability tools?** → Use pre-built OpenTelemetry exporters (Langfuse, LangSmith, etc.)
- **Just getting started?** → Use Raw Exporter with console or file output
- **Integrating with custom HTTP API?** → Use Span Exporter
- **Need custom OTLP service?** → Create simple config wrapper around `OTLPSpanAdapterExporter`
- **Need complex business logic with state tracking?** → Advanced Custom Exporter with custom processors

#### Raw Exporters

Process raw `IntermediateStep` events directly:

- **Use case**: Simple file logging, custom event processing
- **Base class**: `RawExporter`
- **Data flow**: `IntermediateStep` → [Processing Pipeline] → `OutputT` → Export

#### Span Exporters

Convert events into spans with lifecycle management:

- **Use case**: Distributed tracing, span-based observability
- **Base class**: `SpanExporter`
- **Data flow**: `IntermediateStep` → `Span` → [Processing Pipeline] → `OutputT` → Export

#### OpenTelemetry Exporters

Specialized for OpenTelemetry-compatible services with many pre-built options:

- **Use case**: OTLP-compatible backends, standard observability tools
- **Base class**: `OtelSpanExporter`
- **Data flow**: `IntermediateStep` → `Span` → [Processing Pipeline] → `OtelSpan` → Export
- **Pre-built integrations**: Langfuse, LangSmith, OpenTelemetry Collector, Patronus, Galileo, Phoenix, RagaAI, Weave

#### Advanced Custom Exporters

Advanced exporters for complex analytics pipelines with state management:

- **Use case**: Complex business logic, stateful data processing, multi-system integrations
- **Base class**: `ProcessingExporter` with custom processors and advanced features
- **Data flow**: `IntermediateStep` → `InputT` → [Enrichment Pipeline] → `OutputT` → Export
- **Key features**: Circuit breakers, dead letter queues, state tracking, custom transformations, performance monitoring

> **Note**: This is a high-complexity pattern. See the [Advanced Custom Exporters](#advanced-custom-exporters) section in Advanced Features for detailed implementation examples.

**Note**: All exporters support optional processing pipelines that can transform, filter, batch, or aggregate data before export. Common processors include batching for efficient transmission, filtering for selective export, and format conversion for compatibility with different backends.

## Pre-Built Telemetry Exporters

Before creating a custom exporter, check if your observability service is already supported:

### Available Integrations

| Service | Type | Installation | Configuration |
|---------|------|-------------|---------------|
| **File** | `file` | `pip install aiqtoolkit` | local file or directory |
| **Langfuse** | `langfuse` | `pip install aiqtoolkit[opentelemetry]` | endpoint + API keys |
| **LangSmith** | `LangSmith` | `pip install aiqtoolkit[opentelemetry]` | endpoint + API key |
| **OpenTelemetry Collector** | `otelcollector` | `pip install aiqtoolkit[opentelemetry]` | endpoint + headers |
| **Patronus** | `patronus` | `pip install aiqtoolkit[opentelemetry]` | endpoint + API key |
| **Galileo** | `galileo` | `pip install aiqtoolkit[opentelemetry]` | endpoint + API key |
| **Phoenix** | `phoenix` | `pip install aiqtoolkit[phoenix]` | endpoint |
| **RagaAI/Catalyst** | `catalyst` | `pip install aiqtoolkit[ragaai]` | API key + project |
| **Weave** | `weave` | `pip install aiqtoolkit[weave]` | project name |

### Simple Configuration Example

```yaml
# workflow.yaml
general:
  telemetry:
    tracing:
      langfuse:
        type: langfuse
        endpoint: "https://cloud.langfuse.com/api/public/otel/v1/traces"
        public_key: "${LANGFUSE_PUBLIC_KEY}"
        secret_key: "${LANGFUSE_SECRET_KEY}"
```

> **Most services use OTLP**: If your service supports OpenTelemetry Protocol (OTLP), you can often subclass `OtelSpanExporter` or use the generic `otelcollector` type with appropriate headers.

## Creating a Custom Telemetry Exporter

This section provides detailed guidance for creating production-ready telemetry exporters. If you just want to get started quickly, see the [Quick Start](#quick-start-your-first-telemetry-exporter) section first.

### Step 1: Define the Configuration Class

Create a configuration class that inherits from `TelemetryExporterBaseConfig`:

```python
from pydantic import Field

from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

class CustomTelemetryExporter(TelemetryExporterBaseConfig, name="custom"):
    """A simple custom telemetry exporter for sending traces to a custom service."""

    # Required fields
    endpoint: str = Field(description="The endpoint URL for the custom service")
    api_key: str = Field(description="API key for authentication")
```

> **Tip**: Start with the fields you need and add more as your integration becomes more sophisticated. See the [Common Integration Patterns](#common-integration-patterns) section for practical examples.

### Step 2: Implement the Exporter Class

Choose the appropriate base class based on your needs:

#### Raw Exporter (for simple trace exports)

```python
from aiq.observability.exporter.raw_exporter import RawExporter
from aiq.data_models.intermediate_step import IntermediateStep

class CustomRawExporter(RawExporter[IntermediateStep, IntermediateStep]):
    """A custom raw exporter that processes intermediate steps directly."""

    def __init__(self, endpoint: str, api_key: str, project: str, **kwargs):
        super().__init__(**kwargs)
        # Store configuration
        self.endpoint = endpoint
        self.api_key = api_key
        self.project = project

    async def export_processed(self, item: IntermediateStep):
        """Export the intermediate step to the custom service."""
        # Transform and send data
        payload = {
            "project": self.project,
            "event_type": item.event_type,
            "name": item.payload.name if item.payload else None,
            "timestamp": item.event_timestamp
        }
        # Send to your service (implement _send_to_service method)
        await self._send_to_service(payload)

    async def _cleanup(self):
        """Clean up resources when the exporter is stopped."""
        # Clean up HTTP sessions, file handles, etc.
        await super()._cleanup()
```

#### Span Exporter (for span-based tracing)

```python
from aiq.data_models.span import Span
from aiq.observability.exporter.span_exporter import SpanExporter
from aiq.observability.processor.processor import Processor

class SpanToDictProcessor(Processor[Span, dict]):
    """Processor that transforms Span objects to dictionaries."""

    async def process(self, item: Span) -> dict:
        """Transform a Span object to a dictionary."""
        return {
            "span_id": item.context.span_id if item.context else None,
            "trace_id": item.context.trace_id if item.context else None,
            "parent_span_id": item.context.parent_span_id if item.context else None,
            "name": item.name,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "duration": item.duration,
            "status": item.status,
            "attributes": item.attributes,
            "events": item.events,
            "links": item.links
        }

class CustomSpanExporter(SpanExporter[Span, dict]):
    """A custom span exporter that sends spans to a custom service."""

    def __init__(self, endpoint: str, api_key: str, project: str, **kwargs):
        super().__init__(**kwargs)
        # Store configuration and initialize resources
        self.endpoint = endpoint
        self.api_key = api_key
        self.project = project

        # Add the processor to transform Span to dict
        self.add_processor(SpanToDictProcessor())

    async def export_processed(self, item: dict):
        """Export the processed span to the custom service."""
        # The item is now a dict thanks to SpanToDictProcessor
        payload = {
            "project": self.project,
            "span": item
        }
        # Send to your service
        await self._send_to_service(payload)

    async def _cleanup(self):
        """Clean up resources when the exporter is stopped."""
        # Clean up HTTP sessions, file handles, etc.
        await super()._cleanup()
```

#### OpenTelemetry Exporter (for OTLP compatibility)

> **Note**: OpenTelemetry exporters require the `aiqtoolkit-opentelemetry` subpackage. Install it with:

> ```bash
> pip install aiqtoolkit[opentelemetry]
> ```

For most OTLP-compatible services, use the pre-built `OTLPSpanAdapterExporter`:

```python
from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

# See Pattern 3 in Common Integration Patterns for full example
```

> **Tip**: For complete implementation examples with HTTP sessions, error handling, and cleanup, see the [Common Integration Patterns](#common-integration-patterns) section.
> **Warning**: Always implement `_cleanup()` and call `await super()._cleanup()` to prevent resource leaks. Failure to properly clean up HTTP sessions, file handles, or database connections can cause memory leaks and connection pool exhaustion in production environments.

### Step 3: Register the Exporter

Create a registration function using the `@register_telemetry_exporter` decorator:

```python
import logging

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter

logger = logging.getLogger(__name__)

@register_telemetry_exporter(config_type=CustomTelemetryExporter)
async def custom_telemetry_exporter(config: CustomTelemetryExporter, builder: Builder):
    """Create a custom telemetry exporter."""

    try:
        # Initialize the exporter with configuration
        exporter = CustomSpanExporter(
            endpoint=config.endpoint,
            api_key=config.api_key,
            project=config.project,
            batch_size=config.batch_size,
            timeout=config.timeout,
            retries=config.retries
        )

        # Yield the exporter (async context manager pattern)
        yield exporter

    except Exception as ex:
        logger.error(f"Failed to create custom telemetry exporter: {ex}", exc_info=True)
        raise
```

> **Important**: For plugin-specific imports (like `aiohttp`, OpenTelemetry modules, or other external dependencies), always import them inside the registration function to enable lazy loading. This prevents long startup times when these plugins aren't needed.

### Best Practices for Code Organization

In production code, structure your telemetry exporter as follows:

```python
# my_plugin/exporters.py
import aiohttp

from aiq.data_models.span import Span
from aiq.observability.exporter.span_exporter import SpanExporter

class MyCustomExporter(SpanExporter[Span, dict]):
    """Custom exporter implementation."""

    def __init__(self, endpoint: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = aiohttp.ClientSession()

    async def export_processed(self, item: dict):
        # Implementation here
        pass

    async def _cleanup(self):
        """Clean up resources when the exporter is stopped."""
        # Clean up HTTP sessions, file handles, etc.
        await super()._cleanup()

# my_plugin/register.py
from pydantic import Field

from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.builder.builder import Builder

# Configuration class can be in the same file as registration
class MyTelemetryExporter(TelemetryExporterBaseConfig, name="my_exporter"):
    endpoint: str = Field(description="Service endpoint URL")
    api_key: str = Field(description="API key for authentication")

@register_telemetry_exporter(config_type=MyTelemetryExporter)
async def my_telemetry_exporter(config: MyTelemetryExporter, builder: Builder):
    # Import only when the exporter is actually used
    from .exporters import MyCustomExporter

    yield MyCustomExporter(
        endpoint=config.endpoint,
        api_key=config.api_key
    )
```

**Why this pattern?**

- **Lazy loading**: Plugin dependencies are only loaded when the exporter is used
- **Clean separation**: Business logic is separate from registration
- **Maintainability**: Classes are easier to test and modify when properly organized
- **Performance**: Avoids importing heavy dependencies during application startup

**Note**: Configuration classes are lightweight and can be defined in the same file as registration functions. The separation is primarily for exporter implementation classes that have heavy dependencies.

> **Note**: For OpenTelemetry exporters with custom protocols, see the [Advanced Features](#advanced-features) section for mixin patterns and complex integrations.

### Step 4: Add Processing Pipeline (Optional)

If your exporter needs to transform data before export, add processors to the pipeline. This is especially important when using `SpanExporter[Span, dict]` to convert `Span` objects to dictionaries:

```python
from aiq.data_models.span import Span
from aiq.observability.processor.processor import Processor

class SpanToDictProcessor(Processor[Span, dict]):
    """Processor that transforms Span objects to dictionaries."""

    async def process(self, item: Span) -> dict:
        """Transform a Span object to a dictionary."""
        return {
            "span_id": item.context.span_id if item.context else None,
            "trace_id": item.context.trace_id if item.context else None,
            "parent_span_id": item.context.parent_span_id if item.context else None,
            "name": item.name,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "duration": item.duration,
            "status": item.status,
            "attributes": item.attributes,
            "events": item.events
        }

class CustomFieldProcessor(Processor[dict, dict]):
    """Processor that adds custom fields to the data."""

    async def process(self, item: dict) -> dict:
        """Add custom fields to the dictionary."""
        return {
            **item,
            "custom_field": self._extract_custom_data(item),
            "processed_at": self._get_current_timestamp()
        }

    def _extract_custom_data(self, item):
        """Extract custom data from the item."""
        # Add custom transformation logic
        return item.get("attributes", {}).get("custom", {})

    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

# Add processors to your exporter
class CustomSpanExporter(SpanExporter[Span, dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add processors to the pipeline (they run in order)
        self.add_processor(SpanToDictProcessor())  # First: Span -> dict
        self.add_processor(CustomFieldProcessor())  # Second: add custom fields
```

**Common processor patterns:**

- **Span to dict transformation**: Convert `Span` objects to dictionaries
- **Field filtering**: Remove sensitive or unnecessary fields
- **Field transformation**: Convert timestamps, normalize data formats
- **Custom enrichment**: Add metadata, context, or computed fields

### Step 5: Configure in Workflow

Once registered, configure your telemetry exporter in your workflow configuration:

```yaml
# workflow.yaml
general:
  telemetry:
    tracing:
      custom_exporter:
        type: custom
        endpoint: "https://api.custom-service.com/traces"
        api_key: "${CUSTOM_API_KEY}"
```

> **Next Steps**: You now have a complete custom telemetry exporter! For real-world implementation examples, see the [Common Integration Patterns](#common-integration-patterns) section. For advanced features like concurrent execution and performance optimization, see the [Advanced Features](#advanced-features) section.

## Common Integration Patterns

These patterns show example exporter implementations. When implementing these in your own registration functions, remember to move plugin-specific imports (like `aiohttp`, OpenTelemetry modules) inside the registration function for lazy loading.

### Pattern 1: HTTP API with Authentication

Most observability services use HTTP APIs with token authentication:

```python
import aiohttp

from aiq.data_models.span import Span
from aiq.observability.exporter.span_exporter import SpanExporter
from aiq.observability.processor.processor import Processor

class SpanToDictProcessor(Processor[Span, dict]):
    """Processor that transforms Span objects to dictionaries."""

    async def process(self, item: Span) -> dict:
        """Transform a Span object to a dictionary."""
        return {
            "span_id": item.context.span_id if item.context else None,
            "trace_id": item.context.trace_id if item.context else None,
            "name": item.name,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "attributes": item.attributes
        }

class HTTPServiceExporter(SpanExporter[Span, dict]):
    def __init__(self, endpoint: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.session = aiohttp.ClientSession()

        # Add processor to transform Span to dict
        self.add_processor(SpanToDictProcessor())

    async def export_processed(self, item: dict):
        # item is now a dict thanks to SpanToDictProcessor
        async with self.session.post(
            self.endpoint,
            json=item,
            headers=self.headers
        ) as response:
            response.raise_for_status()

    async def _cleanup(self):
        """Clean up HTTP session."""
        await self.session.close()
        await super()._cleanup()
```

### Pattern 2: File-based Export

For local development and debugging:

```python
import asyncio
import aiofiles

from aiq.observability.exporter.raw_exporter import RawExporter
from aiq.observability.processor.intermediate_step_serializer import IntermediateStepSerializer

class FileExporter(RawExporter[IntermediateStep, str]):
    def __init__(self, filepath: str, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.lock = asyncio.Lock()
        self.add_processor(IntermediateStepSerializer())

    async def export_processed(self, item: str):
        async with self._lock:
            async with aiofiles.open(self._current_file_path, mode="a") as f:
                f.write(item + '\n')
```

### Pattern 3: Quick OpenTelemetry Integration

For standard OTLP services, use the pre-built adapter:

```python
@register_telemetry_exporter(config_type=MyTelemetryExporter)
async def my_telemetry_exporter(config: MyTelemetryExporter, builder: Builder):
    # Import inside the function for lazy loading
    from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers={"Authorization": f"Bearer {config.api_key}"},
        batch_size=config.batch_size
    )
```

> **Summary**: You now have three proven patterns for telemetry integration:

> - **Pattern 1 (HTTP API)**: Most common for cloud services and APIs
> - **Pattern 2 (File Export)**: Perfect for development and debugging
> - **Pattern 3 (OTLP)**: Use when your service supports OpenTelemetry standards
>
> For basic integrations, these patterns cover 90% of use cases. Continue to Advanced Features only if you need concurrent execution, high-performance batching, or advanced error handling.

## Advanced Features

This section covers advanced topics for production-ready telemetry exporters. Choose the sections relevant to your use case:

- **[Concurrent Execution](#isolated-attributes-for-concurrent-execution)**: Required for multi-user or multi-workflow applications
- **[Custom OpenTelemetry Protocols](#custom-opentelemetry-protocols)**: Advanced OpenTelemetry integration patterns
- **[Performance Optimization](#performance-optimization)**: Batching, connection management, and efficiency
- **[Reliability](#error-handling-and-retries)**: Error handling, retries, and resilience
- **[Advanced Custom Exporters](#advanced-custom-exporters)**: State-aware processing, data warehouses, and complex pipelines

### Concurrent Execution

#### Isolated Attributes for Concurrent Execution

> **Note**: If you're only running one workflow at a time, you can skip this section. However, if your application runs multiple concurrent workflows or serves multiple users simultaneously, proper isolation is critical to prevent data corruption and race conditions.

When multiple workflows run simultaneously, each needs its own isolated exporter state. NeMo Agent toolkit provides `IsolatedAttribute` to handle this automatically.

#### The Problem

Without isolation, concurrent workflows would share the same exporter instance, leading to:

- Mixed-up trace data between workflows
- Race conditions in processing queues
- Incorrect metrics and task tracking

#### The Solution: IsolatedAttribute

`IsolatedAttribute` creates separate state for each workflow while sharing expensive resources:

```python
from aiq.data_models.span import Span
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.observability.exporter.span_exporter import SpanExporter

class MyExporter(SpanExporter[Span, dict]):

    # Isolated mutable state per workflow (safe)
    _processing_queue: IsolatedAttribute[deque] = IsolatedAttribute(deque)
    _metrics: IsolatedAttribute[dict] = IsolatedAttribute(dict)

    def __init__(self, endpoint: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        # Instance-level resources - each exporter gets its own
        self.endpoint = endpoint
        self.session = aiohttp.ClientSession()
        self.headers = {"Authorization": f"Bearer {api_key}"}
```

**Built-in Usage**: The base exporter classes already use `IsolatedAttribute` for core functionality:

- `BaseExporter` uses it for `_tasks`, `_ready_event`, and `_shutdown_event`
- `SpanExporter` uses it for `_outstanding_spans`, `_span_stack`, and `_metadata_stack`

This ensures that each isolated instance has its own task tracking and span lifecycle management.

#### Usage in Exporters

```python
import uuid
import aiohttp
from collections import deque

from aiq.data_models.span import Span
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.observability.exporter.span_exporter import SpanExporter

class MyCustomExporter(SpanExporter[Span, dict]):
    """Custom exporter with isolated state management."""

    # Isolated mutable state per workflow (safe)
    _processing_queue: IsolatedAttribute[deque] = IsolatedAttribute(deque)
    _active_requests: IsolatedAttribute[set] = IsolatedAttribute(set)
    _export_metrics: IsolatedAttribute[dict] = IsolatedAttribute(dict)

    def __init__(self, endpoint: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        # Store configuration as instance variables
        self.endpoint = endpoint
        self.api_key = api_key

        # Create HTTP client and headers per instance
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def export_processed(self, item: dict):
        """Export with isolated state tracking."""
        # Use isolated attributes for mutable state
        self._processing_queue.append(item)
        request_id = str(uuid.uuid4())
        self._active_requests.add(request_id)

        try:
            # Use instance HTTP client and headers
            async with self.session.post(
                self.endpoint,
                json=item,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    self._export_metrics['success'] = self._export_metrics.get('success', 0) + 1
                else:
                    self._export_metrics['failure'] = self._export_metrics.get('failure', 0) + 1

        finally:
            self._active_requests.discard(request_id)
            if self._processing_queue:
                self._processing_queue.popleft()

    async def _cleanup(self):
        """Clean up HTTP session."""
        await self.session.close()
        await super()._cleanup()
```

#### How Isolation Works

When `create_isolated_instance()` is called, the `IsolatedAttribute` descriptor automatically:

1. **Shares expensive resources**: HTTP clients, authentication headers, etc.
2. **Isolates mutable state**: Each instance gets its own queue, metrics, tracking sets
3. **Maintains thread safety**: No locks needed for concurrent access

```python
# Original exporter
exporter1 = MyCustomExporter("https://api.service1.com")
exporter1._processing_queue.append("item1")
exporter1._export_metrics['success'] = 5

# Create isolated instance
context_state = AIQContextState.get()
exporter2 = exporter1.create_isolated_instance(context_state)

# Isolated state - each has independent data
assert len(exporter1._processing_queue) == 1  # Has "item1"
assert len(exporter2._processing_queue) == 0  # Empty queue
assert exporter1._export_metrics['success'] == 5  # Original metrics
assert len(exporter2._export_metrics) == 0  # Fresh metrics

# Shared resources - same HTTP session
assert exporter1.session is exporter2.session  # Same session
```

#### Best Practices for IsolatedAttribute

**Use IsolatedAttribute for:**

- Task tracking sets
- Processing queues
- Metrics dictionaries
- Event tracking state
- Temporary buffers
- Request counters

**Don't use IsolatedAttribute for:**

- HTTP clients (expensive to create)
- Authentication tokens
- Configuration settings
- Database connections
- Logger instances

 **Example with Common Patterns:**

 ```python
from collections import deque

import aiohttp

from aiq.data_models.span import Span
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.observability.exporter.span_exporter import SpanExporter

class BatchingExporter(SpanExporter[Span, dict]):
    """Exporter demonstrating common IsolatedAttribute patterns."""

    # Isolated mutable state per workflow (safe)
    _batch_queue: IsolatedAttribute[deque] = IsolatedAttribute(deque)
    _flush_timer: IsolatedAttribute[dict] = IsolatedAttribute(dict)
    _statistics: IsolatedAttribute[dict] = IsolatedAttribute(
        lambda: {"batches_sent": 0, "items_processed": 0, "errors": 0}
    )

    def __init__(self, batch_size: int = 100, endpoint: str = "https://your-service.com/api/spans", **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.endpoint = endpoint

        # Define headers once during initialization
        self.headers = {
            "Content-Type": "application/json"
        }

        # Create HTTP session once and reuse it
        import aiohttp
        self.session = aiohttp.ClientSession()

    async def export_processed(self, item: dict):
        """Export with batching and isolated state."""
        # Add to isolated batch queue
        self._batch_queue.append(item)
        self._statistics['items_processed'] += 1

        # Flush if batch is full
        if len(self._batch_queue) >= self.batch_size:
            await self._flush_batch()

    async def _flush_batch(self):
        """Flush batch with isolated state management."""
        if not self._batch_queue:
            return

        # Create batch from isolated queue
        batch = list(self._batch_queue)
        self._batch_queue.clear()

        try:
            # Send batch directly with proper error handling
            await self._send_batch(batch)
            self._statistics['batches_sent'] += 1
        except Exception as e:
            self._statistics['errors'] += 1
            # In production, you might want to retry or use a dead letter queue
            raise

    async def _send_batch(self, batch: list[dict]):
        """Send batch to the service."""
        payload = {"spans": batch}

        # Use the reusable session and headers
        async with self.session.post(
            self.endpoint,
            json=payload,
            headers=self.headers
        ) as response:
            response.raise_for_status()

    async def _cleanup(self):
        """Clean up HTTP session."""
        if hasattr(self, 'session') and self.session:
            await self.session.close()
        await super()._cleanup()
```

### Custom OpenTelemetry Protocols

**Use Case**: When you need to integrate with an OpenTelemetry-compatible service that requires custom authentication, headers, or data transformation.

For OpenTelemetry exporters with custom protocols, create a simple mixin that handles authentication and HTTP transport:

```python
# In production, define these classes in a separate module (e.g., exporters.py)
import aiohttp

from aiq.plugins.opentelemetry.otel_span import OtelSpan

class CustomProtocolMixin:
    """Simple mixin for custom authentication and HTTP transport."""

    def __init__(self, *args, endpoint: str, api_key: str, **kwargs):
        """Initialize the custom protocol mixin."""
        self.endpoint = endpoint
        self.api_key = api_key

        # Define headers once during initialization
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.session = aiohttp.ClientSession()
        super().__init__(*args, **kwargs)

    async def export_otel_spans(self, spans: list[OtelSpan]):
        """Export spans using the custom protocol."""

        # Simple payload - send spans with minimal wrapping
        payload = {
            "spans": [
                {
                    "name": span.name,
                    "span_id": span.get_span_context().span_id,
                    "trace_id": span.get_span_context().trace_id,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes) if span.attributes else {}
                }
                for span in spans
            ]
        }

        # Send to service with custom headers
        async with self.session.post(
            self.endpoint,
            json=payload,
            headers=self.headers
        ) as response:
            response.raise_for_status()

    async def _cleanup(self):
        """Clean up HTTP session."""
        await self.session.close()
        await super()._cleanup()

# In production, you would define this in a separate module and import OtelSpanExporter there
# For example: from aiq.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter
# class CustomServiceExporter(CustomProtocolMixin, OtelSpanExporter):
#     """Simple exporter combining custom protocol with OpenTelemetry span processing."""
#     def __init__(self, endpoint: str, api_key: str, **kwargs):
#         super().__init__(endpoint=endpoint, api_key=api_key, **kwargs)

@register_telemetry_exporter(config_type=CustomTelemetryExporter)
async def custom_telemetry_exporter(config: CustomTelemetryExporter, builder: Builder):
    """Create a custom telemetry exporter using the mixin pattern."""

    # In production, import your exporter classes from a separate module:
    # from .exporters import CustomServiceExporter

    # For this example, we'll create a simple combined class here
    from aiq.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter

    class CustomServiceExporter(CustomProtocolMixin, OtelSpanExporter):
        """Simple exporter combining custom protocol with OpenTelemetry span processing."""
        def __init__(self, endpoint: str, api_key: str, **kwargs):
            super().__init__(endpoint=endpoint, api_key=api_key, **kwargs)

    yield CustomServiceExporter(
        endpoint=config.endpoint,
        api_key=config.api_key
    )
```

> **For Complex Transformations**: This example shows basic field mapping. If you need complex data transformations, filtering, or enrichment, consider using dedicated [Processor classes](#step-4-add-processing-pipeline-optional) instead of inline transformations. Processors are reusable, testable, and can be chained for complex pipelines.

### Performance Optimization

#### Batching Support

**Use Case**: High-throughput applications generating hundreds or thousands of traces per second.

**Conceptual Flow:**
```
1. Configure BatchingProcessor with size/time limits
2. Add processor to exporter pipeline
3. Handle both individual items and batches in export_processed()
4. Transform data to target format
5. Send HTTP request with batched payload
```

**Implementation Pattern:**
```python
class BatchingExporter(RawExporter[IntermediateStep, IntermediateStep]):
    def __init__(self, endpoint, api_key, batch_size=100, flush_interval=5.0):
        super().__init__()
        # Store connection details
        self.endpoint = endpoint
        self.session = aiohttp.ClientSession()
        self.headers = {"Authorization": f"Bearer {api_key}"}

        # Add batching with size and time triggers
        self.add_processor(BatchingProcessor[IntermediateStep](
            batch_size=batch_size,
            flush_interval=flush_interval
        ))

    async def export_processed(self, item: IntermediateStep | list[IntermediateStep]):
        # Handle both single items and batches from processor
        items = item if isinstance(item, list) else [item]
        await self._send_batch(items)

    async def _send_batch(self, items: list[IntermediateStep]):
        # Transform to target format
        payload = {"events": [self._transform_item(item) for item in items]}

        # Send to service
        async with self.session.post(self.endpoint, json=payload, headers=self.headers) as response:
            response.raise_for_status()
```

**Key Features of BatchingProcessor:**

- **Size-based batching**: Flushes when `batch_size` items are accumulated
- **Time-based batching**: Flushes after `flush_interval` seconds
- **Auto-wired callbacks**: Callbacks automatically set up when added to exporter
- **Shutdown safety**: Processes all queued items during cleanup
- **Overflow handling**: Configurable drop behavior when queue is full
- **Statistics**: Built-in metrics for monitoring performance

**Configuration Options:**

```python
BatchingProcessor[T](
    batch_size=100,           # Items per batch
    flush_interval=5.0,       # Seconds between flushes
    max_queue_size=1000,      # Maximum queue size
    drop_on_overflow=False,   # Drop items vs. force flush
    shutdown_timeout=10.0     # Shutdown timeout
)
```

### Reliability

#### Error Handling and Retries

**Use Case**: Production environments where network issues or service outages are common.

Implement robust error handling:

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientExporter(SpanExporter[Span, dict]):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def export_processed(self, item: dict):
        """Export with retry logic."""
        try:
            await self._export_to_service(item)
        except Exception as ex:
            logger.warning(f"Export failed, retrying: {ex}")
            raise
```

#### Connection Management

**Use Case**: Long-running services that need optimized connection pooling and lifecycle management.

**Conceptual Flow:**
```
1. Override start() method with async context manager
2. Configure connection pool settings (limits, timeouts, DNS cache)
3. Create HTTP session with optimized settings
4. Assign session to instance for use in export_processed()
5. Automatically clean up session when exporter stops
```

**Implementation Pattern:**
```python
class ConnectionManagedExporter(SpanExporter[Span, dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = None

    @asynccontextmanager
    async def start(self):
        # Configure connection pool
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30)

        # Create managed session
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.session = session
            async with super().start():
                yield  # Session automatically closed when context exits
```

### Advanced Custom Exporters

Advanced Custom Exporters are for complex scenarios that require enterprise-grade patterns like circuit breakers, dead letter queues, stateful processing, and multi-backend coordination.

> **For most use cases**, the simpler OpenTelemetry, Span, or Raw exporter patterns are sufficient and recommended. Consider this complexity level only when you have specific enterprise requirements that cannot be met with standard patterns.

## Testing Your Exporter

Create tests for your exporter:

```python
import pytest
from unittest.mock import AsyncMock, patch
from aiq.data_models.intermediate_step import IntermediateStep

@pytest.fixture
def custom_exporter():
    return CustomSpanExporter(
        endpoint="https://test.example.com",
        api_key="test-key",
        project="test-project"
    )

@pytest.mark.asyncio
async def test_export_processed(custom_exporter):
    """Test that export_processed sends data correctly."""
    with patch.object(custom_exporter, '_send_to_service', new_callable=AsyncMock) as mock_send:
        test_item = {"span_id": "123", "name": "test_span"}

        await custom_exporter.export_processed(test_item)

        mock_send.assert_called_once()
        sent_data = mock_send.call_args[0][0]
        assert sent_data["project"] == "test-project"
        assert sent_data["span_id"] == "123"

def test_isolated_attributes():
    """Test that isolated attributes work correctly across instances."""
    from aiq.builder.context import AIQContextState

    # Create original exporter
    exporter1 = CustomSpanExporter(
        endpoint="https://test.example.com",
        api_key="test-key",
        project="test-project"
    )

    # Add data to first exporter's isolated attributes
    exporter1._processing_queue.append("item1")
    exporter1._active_requests.add("request1")
    exporter1._export_metrics["success"] = 5

    # Create isolated instance
    context_state = AIQContextState.get()
    exporter2 = exporter1.create_isolated_instance(context_state)

    # Add different data to second exporter
    exporter2._processing_queue.append("item2")
    exporter2._active_requests.add("request2")
    exporter2._export_metrics["failure"] = 3

    # Test isolation - each exporter has its own state
    assert len(exporter1._processing_queue) == 1
    assert "item1" in exporter1._processing_queue
    assert "item2" not in exporter1._processing_queue

    assert len(exporter2._processing_queue) == 1
    assert "item2" in exporter2._processing_queue
    assert "item1" not in exporter2._processing_queue

    # Test independent metrics
    assert exporter1._export_metrics["success"] == 5
    assert "failure" not in exporter1._export_metrics
    assert exporter2._export_metrics["failure"] == 3
    assert "success" not in exporter2._export_metrics

    # Test request tracking isolation
    assert "request1" in exporter1._active_requests
    assert "request2" not in exporter1._active_requests
    assert "request2" in exporter2._active_requests
    assert "request1" not in exporter2._active_requests
```

## Best Practices

### Performance Considerations
- Use async operations for all I/O
- Implement batching for high-throughput scenarios
- Use connection pooling for HTTP requests
- Consider memory usage with large batches
- Use `IsolatedAttribute` for mutable state in concurrent execution
- Call `create_isolated_instance()` when running multiple workflows concurrently
- Share expensive resources (HTTP clients, auth) across isolated instances

### Error Handling
- Implement retry logic with exponential backoff
- Log errors appropriately without exposing sensitive data
- Gracefully handle service unavailability
- Provide meaningful error messages

### Resource Management
- **Always implement `_cleanup()`**: Override this method to clean up resources like HTTP sessions, file handles, database connections
- **Call parent cleanup**: Always call `await super()._cleanup()` in your override
- **Automatic lifecycle**: The base class calls `_cleanup()` during shutdown - no manual calls needed
- **Handle cleanup errors**: Wrap cleanup operations in try/except blocks to prevent shutdown failures

### Security

> **Warning**: Telemetry data may contain sensitive information from workflow executions. Never log API keys, credentials, or PII in trace data. Always use environment variables for secrets and validate/sanitize data before transmission.

- Never log sensitive data like API keys
- Use environment variables for credentials
- Implement proper authentication
- Validate input data

### Monitoring
- Include metrics for export success/failure rates
- Monitor batch sizes and processing times
- Add health checks for external services
- Log important events for debugging

## Troubleshooting

### Common Issues

**Exporter not found**: Ensure your exporter is properly registered and the module is imported.

**Connection errors**: Check endpoint URLs, authentication, and network connectivity.

**Data format issues**: Verify that your data transformation matches the expected format.

**Performance problems**: Review batching settings and connection pool configurations.

**Concurrent execution issues**: Ensure mutable state uses `IsolatedAttribute` and expensive resources are shared properly.

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("aiq.observability").setLevel(logging.DEBUG)
```

### FAQ

**Q: Which exporter type should I use?**

- **Raw Exporter**: For simple file/console output or custom processing
- **Span Exporter**: For HTTP APIs and services that don't support OTLP but require a span-based trace
- **OpenTelemetry Exporter**: For OTLP-compatible services (recommended for new integrations)

**Q: How do I handle authentication?**

- Use environment variables for credentials: `api_key: str = Field(default="", description="API key from MYSERVICE_API_KEY")`
- Environment variables can be configured directly in the workflow YAML configuration file through [Environment Variable Interpolation](../workflows/workflow-configuration.md#environment-variable-interpolation)
- Check environment variables in registration: `api_key = config.api_key or os.environ.get("MYSERVICE_API_KEY")`

**Q: My exporter isn't receiving events. What's wrong?**

- Verify the exporter is registered and imported
- Check your workflow configuration file syntax
- Enable debug logging to see registration messages
- Ensure the exporter type name matches your configuration

**Q: How do I test my exporter?**

- Start with the console exporter pattern from Quick Start
- Use the file exporter pattern to write traces to a local file
- Test with a simple workflow before integrating with external services

## Complete Example

**Implementation Overview:**
```
1. Define Configuration Schema (TelemetryExporterBaseConfig)
   - Endpoint, API key, project settings
   - Use pydantic Field() for validation and description

2. Create Exporter Class (SpanExporter)
   - Initialize HTTP session and headers in __init__
   - Use IsolatedAttribute for concurrent state management
   - Implement export_processed() with error handling
   - Implement _cleanup() for resource management

3. Register with AIQ (register_telemetry_exporter decorator)
   - Create async factory function
   - Instantiate exporter with config values
   - Yield exporter instance
```

Here's a complete example of a custom telemetry exporter:

```python
import logging
from pydantic import Field
import aiohttp
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.observability.exporter.span_exporter import SpanExporter
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.data_models.span import Span

logger = logging.getLogger(__name__)

# Configuration
class ExampleTelemetryExporter(TelemetryExporterBaseConfig, name="example"):
    endpoint: str = Field(description="Service endpoint")
    api_key: str = Field(description="API key")
    project: str = Field(description="Project name")

# Exporter implementation (in production, define this in a separate module)
class ExampleSpanExporter(SpanExporter[Span, dict]):
    # Isolated mutable state
    _request_counter: IsolatedAttribute[dict] = IsolatedAttribute(
        lambda: {"sent": 0, "failed": 0}
    )

    def __init__(self, endpoint: str, api_key: str, project: str, context_state=None):
        super().__init__(context_state=context_state)
        self.endpoint = endpoint
        self.api_key = api_key
        self.project = project

        # HTTP client as instance variable - shared via shallow copy for isolated instances
        # Import here to avoid loading aiohttp unless this exporter is used
        self.session = aiohttp.ClientSession()
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def export_processed(self, item: dict):
        payload = {"project": self.project, "span": item}

        try:
            async with self.session.post(
                self.endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    self._request_counter["sent"] += 1
                else:
                    self._request_counter["failed"] += 1
                    logger.error(f"Export failed: {response.status}")
        except Exception as e:
            self._request_counter["failed"] += 1
            logger.error(f"Export error: {e}")

    async def _cleanup(self):
        """Clean up shared resources."""
        await self.session.close()
        await super()._cleanup()

# Registration
@register_telemetry_exporter(config_type=ExampleTelemetryExporter)
async def example_telemetry_exporter(config: ExampleTelemetryExporter, builder: Builder):
    # In production, import your exporter class from a separate module:
    # from .exporters import ExampleSpanExporter

    exporter = ExampleSpanExporter(
        endpoint=config.endpoint,
        api_key=config.api_key,
        project=config.project
    )

    yield exporter
```

For additional reference examples, refer to the existing exporter implementations in the toolkit source code.

## Next Steps

1. **Explore Examples**: Check the `examples/observability` directory for workflow examples with configured observability settings
2. **Start Simple**: Begin with the Quick Start console exporter example
3. **Explore Supported Telemetry Exporters**: Look at existing exporters in the `packages/` directory
4. **Choose Your Pattern**: Select Raw, Span, or OpenTelemetry based on your needs
5. **Test Locally**: Use file output first, then integrate with your service
6. **Add Advanced Features**: Implement batching, retry logic, and error handling as needed
