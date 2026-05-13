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
See the License for the specific language governing permissions and
limitations under the License.
-->

# Telemetry & Observability

Logging and tracing under `general.telemetry`: OTel collector, Langfuse, LangSmith, Weave, plus a programmatic-API path for writing OTel traces to local files.

NeMo Agent toolkit's telemetry system lives under `general.telemetry` and supports two subsystems: **logging** (structured log output) and **tracing** (distributed traces). Multiple providers can be active simultaneously.

## Logging

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO           # DEBUG, INFO, WARNING, ERROR
      file:
        _type: file
        path: ./logs/workflow.log
        level: DEBUG
```

## Tracing

### OTel Collector (recommended for custom setups)

Install the OTel and profiling extras. Profiling is required to capture LLM and tool call spans — without it, traces only contain top-level workflow spans.

```bash
uv add "nvidia-nat[opentelemetry,profiler]"
```

`nvidia-nat[profiler]` installs `nvidia-nat-eval`, which hooks into `WorkflowBuilder` via LangChain's `register_configure_hook` and automatically injects `LangchainProfilerHandler` into every run. No code changes needed — just having the package installed activates it.

```yaml
general:
  telemetry:
    tracing:
      otel:
        _type: otelcollector
        endpoint: http://localhost:6006/v1/traces
        project: my-project
```

### Langfuse

```yaml
general:
  telemetry:
    tracing:
      langfuse:
        _type: langfuse
        endpoint: ${LANGFUSE_HOST}/api/public/otel/v1/traces
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
```

### LangSmith

```yaml
general:
  telemetry:
    tracing:
      langsmith:
        _type: langsmith
        project: my-agent-project
```

### Weave (Weights & Biases)

```yaml
general:
  telemetry:
    tracing:
      weave:
        _type: weave
        project: my-project
```

Requires `nvidia-nat[weave]` and `WANDB_API_KEY`. Supports automatic PII redaction via `redact_pii: true` (auto-installs `presidio-analyzer`/`presidio-anonymizer`).

### Phoenix (Arize)

```yaml
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://0.0.0.0:6006
        project: my-project
```

Requires `nvidia-nat[phoenix]` and a running Phoenix server (Docker: `arizephoenix/phoenix:13.22`).

### Catalyst (RagaAI)

```yaml
general:
  telemetry:
    tracing:
      catalyst:
        _type: catalyst
        project: my-project
        dataset: my-dataset
```

Requires `nvidia-nat[ragaai]` and three env vars: `CATALYST_ACCESS_KEY`, `CATALYST_SECRET_KEY`, `CATALYST_ENDPOINT`. The `[ragaai]` extra **conflicts with `[strands]` and `[adk]`** — see the conflicts matrix in `SKILL.md`.

### Galileo

```yaml
general:
  telemetry:
    tracing:
      galileo:
        _type: galileo
        project: my-project
        logstream: my-logstream
```

Uses `nvidia-nat[opentelemetry]`. Requires `GALILEO_API_KEY`. Default endpoint: `https://app.galileo.ai/api/galileo/otel/traces` — override via `endpoint:` if self-hosting. Create the Logging project + Log Stream in the Galileo UI before starting the workflow.

### DBNL

```yaml
general:
  telemetry:
    tracing:
      dbnl:
        _type: dbnl
```

Uses `nvidia-nat[opentelemetry]`. Requires self-hosted DBNL deployment plus three env vars: `DBNL_API_URL`, `DBNL_API_TOKEN`, `DBNL_PROJECT_ID`. Create a Trace Ingestion project in DBNL and generate the API token first.

### NVIDIA Data Flywheel

```yaml
general:
  telemetry:
    tracing:
      flywheel:
        _type: data_flywheel_elasticsearch
        client_id: my-client
        index: my-index
        endpoint: https://elasticsearch.example.com
        username: elastic
        password: $ELASTIC_PASSWORD
        batch_size: 100
```

Requires `nvidia-nat[data-flywheel]`. **Currently supports LangChain / LangGraph workflows with `nim` and `openai` LLM providers only.** Captures `LLM_START` events plus tool calls. Use `@track_unregistered_function` to scope custom workloads.

### Dynatrace

Dynatrace is consumed via the OTel Collector path — point an OTel Collector at Dynatrace's OTLP API and have NeMo Agent toolkit export to that collector with `_type: otelcollector` (see the **OTel Collector** section above). The Dynatrace API token needs the `openTelemetryTrace.ingest` scope.

### Patronus

Supported but documented sparsely upstream. See the canonical example in the NeMo-Agent-Toolkit repo: `examples/observability/simple_calculator_observability`.

## OTel via programmatic API

```python
import asyncio
from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.loader import load_config
from nat.plugins.opentelemetry.register import OtelCollectorTelemetryExporter

async def run_workflow(query: str) -> str:
    config = load_config("workflow.yaml")
    async with WorkflowBuilder.from_config(config) as builder:
        await builder.add_telemetry_exporter(
            "otel",
            OtelCollectorTelemetryExporter(
                endpoint="http://localhost:6006/v1/traces",
                project="my-project",
            ),
        )
        workflow = await builder.build()
        async with workflow.run(query) as runner:
            return str(await runner.result())

if __name__ == "__main__":
    print(asyncio.run(run_workflow("Hello")))
```

**Key points:**

- `endpoint` is the OTLP HTTP traces endpoint of any compatible collector.
- `project` is attached as a resource attribute (`service.name`) on all spans.
- Call `add_telemetry_exporter` before `build()` — exporters are wired in during the build phase.

## Writing OTel traces to a local file

If you want to write OTel traces to a file, copy **[`otel_file_exporter.py`](otel_file_exporter.py)** into your project directory. It registers the `otelfile` exporter type and writes one raw `OtelSpan.to_json()` line per span. Make sure opentelemetry and profiling are installed:

```bash
uv add "nvidia-nat[opentelemetry,profiler]"
```

Then in your main.py file, import the exporter:

```python
# main.py
import otel_file_exporter  # noqa: F401 — registers the 'otelfile' exporter type

# ... run your workflow ...
```

```yaml
# workflow.yaml
general:
  telemetry:
    tracing:
      otel_file:
        _type: otelfile
        output_path: traces/trace.jsonl
```
