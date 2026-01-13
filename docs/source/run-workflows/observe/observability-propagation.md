<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Observability Propagation

Observability propagation enables distributed tracing across multiple NeMo Agent toolkit workflows. When a parent workflow calls a child workflow (either via HTTP or programmatically), trace context flows from parent to child. After execution, trace data from child workflows can be aggregated back into the parent's trace.

This feature is essential for:

- Correlating traces across multi-agent architectures
- Debugging distributed workflows with end-to-end visibility
- Monitoring complex agentic systems that call external services

## Architecture Overview

Observability propagation works in two directions:

1. **Parent to Child**: Trace context (trace ID, span ID, workflow run ID) flows from the calling workflow to the called workflow
2. **Child to Parent**: Intermediate steps and trace events from child workflows are aggregated back into the parent's trace

```
┌─────────────────────────────────────────────────────────────────┐
│                      Parent Workflow                            │
│  ┌───────────────┐                      ┌───────────────────┐   │
│  │ Start Request │──── HTTP Headers ───▶│  Child Workflow   │   │
│  │               │    (traceparent,     │                   │   │
│  │               │    X-NAT-Trace-ID)   │                   │   │
│  │               │                      │                   │   │
│  │               │◀── Response with ────│                   │   │
│  │  Merge Steps  │    _trace payload    │                   │   │
│  └───────────────┘                      └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Parent to Child: Propagating Trace Context

There are two ways to pass parent trace context to child workflows:

### Automatic HTTP Header Propagation

When a child workflow is served via the FastAPI front end, trace context is automatically extracted from incoming HTTP headers. The following headers are supported:

| Header | Description |
|--------|-------------|
| `traceparent` | W3C Trace Context header: `00-<trace-id>-<span-id>-<flags>` |
| `X-NAT-Trace-ID` | NAT-specific trace identifier (32-character hex string) |
| `X-NAT-Span-ID` | NAT-specific span identifier for parent-child linking |
| `X-NAT-Workflow-Run-ID` | Stable workflow run identifier (UUID string) |

Legacy headers are also supported for backward compatibility:

| Legacy Header | Description |
|---------------|-------------|
| `workflow-trace-id` | Alternative trace ID header |
| `workflow-run-id` | Alternative workflow run ID header |

#### Building Outbound Headers

To propagate observability context when calling a child workflow via HTTP, use the `inject_observability_headers()` helper function:

```python
import httpx
from nat.runtime.session import inject_observability_headers

async def call_child_workflow(payload: dict) -> dict:
    headers = inject_observability_headers()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://child-workflow:8000/v1/workflow/full",
            json=payload,
            headers=headers
        )
        return response.json()
```

The `inject_observability_headers()` function builds the appropriate headers from the current context state, including:

- `traceparent`: W3C Trace Context format
- `X-NAT-Trace-ID`: 32-character hex trace ID
- `X-NAT-Span-ID`: Current span ID for parent-child linking
- `X-NAT-Workflow-Run-ID`: Workflow run identifier

You can also merge with existing headers:

```python
existing_headers = {"Authorization": "Bearer token123"}
headers = inject_observability_headers(existing_headers)
```

### Explicit Parent ID in Workflow Runner

When using NAT programmatically (not through HTTP), you can explicitly pass parent information to the workflow runner:

```python
from nat.runtime.session import Session

async with session.run(
    message=payload,
    parent_id="parent-invocation-uuid",
    parent_name="parent_function_name"
) as runner:
    result = await runner.result()
```

The `parent_id` and `parent_name` parameters are set on the root `InvocationNode`, establishing the parent-child relationship in the trace hierarchy.

This approach is useful when:

- Calling workflows from within other workflows programmatically
- Building custom orchestration logic
- Integrating NAT workflows into existing applications

## Child to Parent: Aggregating Trace Data

After a child workflow completes, trace data can be returned to the parent workflow for aggregation.

### Automatic Return via `/v1/workflow/full`

The `/v1/workflow/full` endpoint (previously `/v1/workflow/full`) streams raw intermediate steps without step adaptor translations. When `embed_trace_in_response` is enabled in the configuration, the response includes a final `_trace` payload:

```yaml
front_end:
  _type: fastapi
  observability:
    enable_header_propagation: true
    embed_trace_in_response: true
```

The response stream includes:

1. `intermediate_data:` events for each intermediate step
2. `data:` event for the final output
3. `data:` event with the `_trace` payload (when enabled)

Example `_trace` payload:

```json
{
  "_trace": {
    "events": [
      {
        "id": "step-uuid-1",
        "type": "FUNCTION_START",
        "name": "my_function",
        "parent_id": "root",
        "payload": "{...}"
      },
      {
        "id": "step-uuid-1",
        "type": "FUNCTION_END",
        "name": "my_function",
        "parent_id": "root",
        "payload": "{...}"
      }
    ]
  }
}
```

### Streaming with Trace Collection

For the streaming endpoint (`/v1/workflow/stream`), trace data can be collected and returned at the end of the stream using the same configuration:

```yaml
front_end:
  _type: fastapi
  observability:
    embed_trace_in_response: true
```

When enabled, the streaming response includes the `_trace` payload as the final event after all intermediate steps and output chunks.

### Response Headers

Non-streaming responses include the `Observability-Trace-Id` header:

```
Observability-Trace-Id: 0123456789abcdef0123456789abcdef
```

This header provides the trace ID for correlation without requiring trace embedding in the response body.

### Merging Child Trace Events

To incorporate child trace events into the parent workflow's trace, use the `merge_trace_events()` helper function:

```python
from nat.observability.trace_merge import merge_trace_events

# After receiving response from child workflow
response_data = await call_child_workflow(payload)
trace_payload = response_data.get("_trace")

# Convert to IntermediateStep objects
child_steps = merge_trace_events(
    trace_payload,
    default_parent="remote_workflow",
    default_function_name="child_workflow",
    default_function_id="child-uuid"
)

# child_steps is a list[IntermediateStep] that can be pushed to the parent's trace
for step in child_steps:
    context.intermediate_step_manager.push_intermediate_step(step.payload)
```

The `merge_trace_events()` function:

- Parses the `_trace` payload from the child response
- Converts each event into an `IntermediateStep` object
- Handles validation errors gracefully to avoid breaking trace aggregation

## Configuration Reference

The `ObservabilityPropagationConfig` is configured under the `observability` section of the FastAPI front end:

```yaml
front_end:
  _type: fastapi
  observability:
    enable_header_propagation: true  # Accept and inject observability headers
    embed_trace_in_response: false   # Include _trace payload in responses
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_header_propagation` | boolean | `true` | Accept incoming observability headers and propagate context to child requests |
| `embed_trace_in_response` | boolean | `false` | Include the `_trace` payload with all intermediate steps at the end of responses |

## Complete Example

Here is a complete example of a parent workflow calling a child workflow with observability propagation:

```python
import httpx
from nat.builder.context import Context
from nat.runtime.session import inject_observability_headers
from nat.observability.trace_merge import merge_trace_events

async def orchestrator_function(input_message: str) -> str:
    context = Context.get()
    
    # Build headers with observability context
    headers = inject_observability_headers()
    
    # Call child workflow
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://child-service:8000/v1/workflow/full",
            json={"input": input_message},
            headers=headers
        )
        
        # Parse streaming response
        result = None
        trace_payload = None
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "_trace" in data:
                    trace_payload = data["_trace"]
                else:
                    result = data
            elif line.startswith("intermediate_data: "):
                # Process intermediate steps if needed
                pass
    
    # Merge child trace into parent
    if trace_payload:
        child_steps = merge_trace_events(trace_payload)
        for step in child_steps:
            context.intermediate_step_manager.push_intermediate_step(step.payload)
    
    return result.get("output", "")
```

## Custom Aggregation Endpoints

For advanced use cases, you can implement custom trace aggregation by:

1. Collecting trace events from multiple child workflows
2. Implementing your own aggregation logic
3. Storing traces in your preferred observability backend

The `merge_trace_events()` function provides a foundation, but you may need additional logic for:

- Deduplicating events across retries
- Handling partial failures in distributed calls
- Custom trace correlation strategies

::::{note}
The NeMo Agent toolkit does not currently provide a built-in endpoint for cross-workflow trace aggregation. Users who need centralized trace collection should either:

- Use the `embed_trace_in_response` option and aggregate in the parent workflow
- Export traces to a centralized observability backend (such as Phoenix, Jaeger, or another supported exporter)
::::

## Related Topics

- [Observe Workflows](observe.md) - General observability configuration
- [Adding Telemetry Exporters](../../extend/custom-components/telemetry-exporters.md) - Custom exporter development
- [REST API Reference](../../reference/rest-api/api-server-endpoints.md) - API endpoint documentation
