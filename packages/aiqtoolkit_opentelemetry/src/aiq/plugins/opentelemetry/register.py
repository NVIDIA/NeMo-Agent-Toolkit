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
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class LangfuseTelemetryExporter(TelemetryExporterBaseConfig, name="langfuse"):
    """A telemetry exporter to transmit traces to externally hosted langfuse service."""

    endpoint: str = Field(description="The langfuse OTEL endpoint (/api/public/otel/v1/traces)")
    public_key: str = Field(description="The Langfuse public key", default="")
    secret_key: str = Field(description="The Langfuse secret key", default="")
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")


@register_telemetry_exporter(config_type=LangfuseTelemetryExporter)
async def langfuse_telemetry_exporter(config: LangfuseTelemetryExporter, builder: Builder):  # pylint: disable=W0613

    import base64

    from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

    secret_key = config.secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = config.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not secret_key or not public_key:
        raise ValueError("secret and public keys are required for langfuse")

    credentials = f"{public_key}:{secret_key}".encode("utf-8")
    auth_header = base64.b64encode(credentials).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_header}"}

    yield OTLPSpanAdapterExporter(endpoint=config.endpoint, headers=headers)


class LangsmithTelemetryExporter(TelemetryExporterBaseConfig, name="langsmith"):
    """A telemetry exporter to transmit traces to externally hosted langsmith service."""

    endpoint: str = Field(
        description="The langsmith OTEL endpoint",
        default="https://api.smith.langchain.com/otel/v1/traces",
    )
    api_key: str = Field(description="The Langsmith API key", default="")
    project: str = Field(description="The project name to group the telemetry traces.")
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")


@register_telemetry_exporter(config_type=LangsmithTelemetryExporter)
async def langsmith_telemetry_exporter(config: LangsmithTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Langsmith telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

    api_key = config.api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("API key is required for langsmith")

    headers = {"x-api-key": api_key, "Langsmith-Project": config.project}
    yield OTLPSpanAdapterExporter(endpoint=config.endpoint, headers=headers)


class OtelCollectorTelemetryExporter(TelemetryExporterBaseConfig, name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    endpoint: str = Field(description="The otel endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create an OpenTelemetry telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

    yield OTLPSpanAdapterExporter(endpoint=config.endpoint,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)


class PatronusTelemetryExporter(TelemetryExporterBaseConfig, name="patronus"):
    """A telemetry exporter to transmit traces to Patronus service."""

    endpoint: str = Field(description="The Patronus OTEL endpoint")
    api_key: str = Field(description="The Patronus API key", default="")
    project: str = Field(description="The project name to group the telemetry traces.")
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")


@register_telemetry_exporter(config_type=PatronusTelemetryExporter)
async def patronus_telemetry_exporter(config: PatronusTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Patronus telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

    api_key = config.api_key or os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        raise ValueError("API key is required for Patronus")

    headers = {
        "x-api-key": api_key,
        "pat-project-name": config.project,
    }
    yield OTLPSpanAdapterExporter(endpoint=config.endpoint,
                                  headers=headers,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)


class GalileoTelemetryExporter(TelemetryExporterBaseConfig, name="galileo"):
    """A telemetry exporter to transmit traces to externally hosted galileo service."""

    endpoint: str = Field(description="The galileo endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")
    logstream: str = Field(description="The logstream name to group the telemetry traces.")
    api_key: str = Field(description="The api key to authenticate with the galileo service.")
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")


@register_telemetry_exporter(config_type=GalileoTelemetryExporter)
async def galileo_telemetry_exporter(config: GalileoTelemetryExporter, builder: Builder):
    """Create a Galileo telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_adapter_exporter import OTLPSpanAdapterExporter

    headers = {
        "Galileo-API-Key": config.api_key,
        "logstream": config.logstream,
        "project": config.project,
    }

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )
