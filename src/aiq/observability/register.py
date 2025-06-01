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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_logging_method
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.logging import LoggingBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.utils.optional_imports import telemetry_optional_import
from aiq.utils.optional_imports import try_import_phoenix

logger = logging.getLogger(__name__)


class PhoenixTelemetryExporter(TelemetryExporterBaseConfig, name="phoenix"):
    """A telemetry exporter to transmit traces to externally hosted phoenix service."""

    endpoint: str = Field(description="The phoenix endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder):
    """Create a Phoenix telemetry exporter."""
    try:
        # If the dependencies are not installed, a TelemetryOptionalImportError will be raised
        phoenix = try_import_phoenix()  # noqa: F841
        from phoenix.otel import HTTPSpanExporter

        yield HTTPSpanExporter(config.endpoint)
    except ConnectionError as ex:
        logger.warning(
            "Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
            ex,
            exc_info=True,
        )


class LangfuseTelemetryExporter(TelemetryExporterBaseConfig, name="langfuse"):
    """A telemetry exporter to transmit traces to externally hosted langfuse service."""

    endpoint: str = Field(description="The langfuse OTEL endpoint (/api/public/otel/v1/traces)")
    public_key: str = Field(description="The Langfuse public key", default="")
    secret_key: str = Field(description="The Langfuse secret key", default="")


@register_telemetry_exporter(config_type=LangfuseTelemetryExporter)
async def langfuse_telemetry_exporter(config: LangfuseTelemetryExporter, builder: Builder):
    """Create a Langfuse telemetry exporter."""

    import base64

    trace_exporter = telemetry_optional_import("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    secret_key = config.secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = config.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not secret_key or not public_key:
        raise ValueError("secret and public keys are required for langfuse")

    credentials = f"{public_key}:{secret_key}".encode("utf-8")
    auth_header = base64.b64encode(credentials).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_header}"}

    yield trace_exporter.OTLPSpanExporter(endpoint=config.endpoint, headers=headers)


class LangsmithTelemetryExporter(TelemetryExporterBaseConfig, name="langsmith"):
    """A telemetry exporter to transmit traces to externally hosted langsmith service."""

    endpoint: str = Field(
        description="The langsmith OTEL endpoint",
        default="https://api.smith.langchain.com/otel/v1/traces",
    )
    api_key: str = Field(description="The Langsmith API key", default="")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=LangsmithTelemetryExporter)
async def langsmith_telemetry_exporter(config: LangsmithTelemetryExporter, builder: Builder):
    """Create a Langsmith telemetry exporter."""

    trace_exporter = telemetry_optional_import("opentelemetry.exporter.otlp.proto.http.trace_exporter")

    api_key = config.api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("API key is required for langsmith")

    headers = {"x-api-key": api_key, "LANGSMITH_PROJECT": config.project}
    yield trace_exporter.OTLPSpanExporter(endpoint=config.endpoint, headers=headers)


class OtelCollectorTelemetryExporter(TelemetryExporterBaseConfig, name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    endpoint: str = Field(description="The otel endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder):
    """Create an OpenTelemetry telemetry exporter."""

    trace_exporter = telemetry_optional_import("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    yield trace_exporter.OTLPSpanExporter(endpoint=config.endpoint)


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
