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
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_logging_method
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.logging import LoggingBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.utils.optional_imports import OptionalImportError
from aiq.utils.optional_imports import get_opentelemetry
from aiq.utils.optional_imports import get_phoenix

logger = logging.getLogger(__name__)


# Define a dummy SpanExporter for when OpenTelemetry is not available
class DummySpanExporter:

    def export(self, *args, **kwargs):
        pass

    def shutdown(self, *args, **kwargs):
        pass


class PhoenixTelemetryExporter(TelemetryExporterBaseConfig, name="phoenix"):
    """A telemetry exporter to transmit traces to externally hosted phoenix service."""

    endpoint: str = Field(description="The phoenix endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder) -> AsyncIterator[Any]:
    try:
        phoenix = get_phoenix()
        yield phoenix.otel.HTTPSpanExporter(endpoint=config.endpoint)
    except OptionalImportError as e:
        logger.warning("Phoenix not available: %s", e)
        yield DummySpanExporter()
    except ConnectionError as ex:
        logger.error("Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
                     ex,
                     exc_info=True)
    except Exception as ex:
        logger.error("Error in Phoenix telemetry Exporter\n %s", ex, exc_info=True)


class OtelCollectorTelemetryExporter(TelemetryExporterBaseConfig, name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    endpoint: str = Field(description="The otel endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder) -> AsyncIterator[Any]:
    try:
        opentelemetry = get_opentelemetry()
        yield opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter(endpoint=config.endpoint)
    except OptionalImportError as e:
        logger.warning("OpenTelemetry not available: %s", e)
        yield DummySpanExporter()
    except Exception as ex:
        logger.error("Error in OTel telemetry Exporter\n %s", ex, exc_info=True)
        yield DummySpanExporter()


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
