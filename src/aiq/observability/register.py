# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
# import copy # Not strictly needed if we only use dict.copy()
# import types # Not needed
from typing import Optional, Sequence
from pydantic import Field
import json # Keep json import

from opentelemetry.sdk.trace import ReadableSpan, Event, InstrumentationScope
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
import opentelemetry.semconv_ai as ot

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_logging_method
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.logging import LoggingBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

# Removed incorrect import
# import opentelemetry.semconv_ai as ot
# from opentelemetry.sdk.trace import ReadableSpan # Already imported above
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult # Already imported above

logger = logging.getLogger(__name__)


def get_wandb_api_key(config_api_key: Optional[str] = None) -> Optional[str]:
    """
    Get the W&B API key from various sources in order of priority:
    1. Config provided key
    2. WANDB_API_KEY environment variable
    Returns:
        The API key if found, None otherwise
    """
    if config_api_key:
        return config_api_key
    # Check environment variable
    env_api_key = os.environ.get("WANDB_API_KEY")
    if env_api_key:
        return env_api_key
    return None


class WeaveTelemetryExporter(TelemetryExporterBaseConfig, name="weave"):
    """A telemetry exporter to transmit traces to Weights & Biases Weave using OpenTelemetry."""
    entity: str = Field(description="The W&B entity/organization.")
    project: str = Field(description="The W&B project name.")
    api_key: Optional[str] = Field(
        default=None,
        description="Your W&B API key for authentication. If not provided, will look for WANDB_API_KEY environment variable."
    )
    endpoint: Optional[str] = Field(
        default="https://trace.wandb.ai/otel/v1/traces",
        description="The Weave OTEL endpoint to export telemetry traces. If not provided, will use the default Weave endpoint."
    )


class AgentIQToWeaveExporter(SpanExporter):
    """
    A wrapper around the real OTLPSpanExporter that renames or reshapes
    AgentIQ attributes so that Weave sees them as 'inputs' and 'outputs'.
    """

    def __init__(self, wrapped_exporter: SpanExporter):
        self._wrapped_exporter = wrapped_exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            # Create a mutable copy of attributes
            original_attributes = dict(span.attributes) if span.attributes else {}
            new_attributes = original_attributes.copy()

            # --- Input Transformation ---
            prompts = {}
            # keys that start with "input"
            inputs = {k: v for k, v in original_attributes.items() if k.startswith("input.")}
            if inputs:
                if inputs["input.mime_type"] == "application/json":
                    inputs = json.loads(inputs["input.value"])

                    new_attributes[ot.SpanAttributes.LLM_PROMPTS] = inputs
                    print(f"new_attributes: {new_attributes}")

            span._attributes = new_attributes

        return self._wrapped_exporter.export(spans)

    def shutdown(self):
        return self._wrapped_exporter.shutdown()


@register_telemetry_exporter(config_type=WeaveTelemetryExporter)
async def weave_telemetry_exporter(config: WeaveTelemetryExporter, builder: Builder):
    import base64
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    class NoOpSpanExporter:
        def export(self, spans):
            return None
        
        def shutdown(self):
            return None

    api_key = get_wandb_api_key(config.api_key)
    if not api_key:
        logger.error("W&B API key not found. Please provide it in the config or set WANDB_API_KEY environment variable.")
        yield NoOpSpanExporter()
        return

    try:
        auth = base64.b64encode(f"api:{api_key}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "project_id": f"{config.entity}/{config.project}"
        }
        # Create and yield the OTLP HTTP exporter
        yield AgentIQToWeaveExporter(
            OTLPSpanExporter(
                endpoint=config.endpoint,
                headers=headers
            )
        )
    except Exception as ex:
        logger.error("Error in Weave telemetry Exporter\n %s", ex, exc_info=True)
        yield NoOpSpanExporter()


class PhoenixTelemetryExporter(TelemetryExporterBaseConfig, name="phoenix"):
    """A telemetry exporter to transmit traces to externally hosted phoenix service."""

    endpoint: str = Field(description="The phoenix endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


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


class OtelCollectorTelemetryExporter(TelemetryExporterBaseConfig, name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    endpoint: str = Field(description="The otel endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder):

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    yield OTLPSpanExporter(endpoint=config.endpoint)


class ConsoleLoggingMethod(LoggingBaseConfig, name="console"):
    """A logger to write runtime logs to the console."""

    level: str = Field(description="The logging level of console logger.")


@register_logging_method(config_type=ConsoleLoggingMethod)
async def console_logging_method(config: ConsoleLoggingMethod, builder: Builder):
    """
        Build and return a StreamHandler for console-based logging.
        """
    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    yield handler


class FileLoggingMethod(LoggingBaseConfig, name="file"):
    """A logger to write runtime logs to a file."""

    path: str = Field(description="The file path to save the logging output.")
    level: str = Field(description="The logging level of file logger.")


@register_logging_method(config_type=FileLoggingMethod)
async def file_logging_method(config: FileLoggingMethod, builder: Builder):
    """
        Build and return a FileHandler for file-based logging.
        """
    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.FileHandler(filename=config.path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    yield handler