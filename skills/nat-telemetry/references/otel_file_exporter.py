"""OTel file exporter — writes raw OtelSpan JSON, one span per line.

Copy to your project directory and import before WorkflowBuilder.from_config():

    import otel_file_exporter  # noqa: F401 — registers the 'otelfile' exporter type

Then add to your workflow YAML:

    general:
      telemetry:
        tracing:
          otel_file:
            _type: otelfile
            output_path: traces/trace.jsonl
"""

import asyncio
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter


class OtelFileExporter(OtelSpanExporter):
    """Writes each OtelSpan as a single JSON line using OtelSpan.to_json()."""

    def __init__(self, output_path: str, endpoint: str | None = None, **kwargs):
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("flush_interval", 0.1)
        super().__init__(**kwargs)
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("")  # truncate at start of each run

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:

        def _write():
            with open(self._path, "a") as f:
                for span in spans:
                    f.write(span.to_json(indent=None) + "\n")

        await asyncio.to_thread(_write)


class OtelFileTelemetryExporterConfig(TelemetryExporterBaseConfig, name="otelfile"):
    output_path: str = Field(description="Path to write raw OtelSpan JSON traces.")


@register_telemetry_exporter(config_type=OtelFileTelemetryExporterConfig)
async def otel_file_exporter(config: OtelFileTelemetryExporterConfig, builder: Builder):
    yield OtelFileExporter(output_path=config.output_path)
