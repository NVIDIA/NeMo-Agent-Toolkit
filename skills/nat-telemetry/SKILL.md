---
name: nat-telemetry
description: Use when adding, configuring, or troubleshooting NeMo Agent toolkit logging, tracing, telemetry exporters, OpenTelemetry, Langfuse, LangSmith, Weave, Phoenix, profiling, or observability provider integrations.
metadata:
  version: "0.1.0"
  status: initial
---

# NeMo Agent toolkit Telemetry

Use this skill for workflow observability, tracing, logging, and profiling.

## Workflow

1. Discover registered logging and tracing exporters:

```bash
uv run nat info components -t logging
uv run nat info components -t tracing
```

2. Prefer built-in provider integrations when available.
3. Use file or console exporters for deterministic local debugging.
4. For custom exporters, adapt `references/otel_file_exporter.py` and nearby toolkit exporter code.

## References

- `references/telemetry.md`
- `references/otel_file_exporter.py`
