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


@register_telemetry_exporter(config_type=LangfuseTelemetryExporter)
async def langfuse_telemetry_exporter(config: LangfuseTelemetryExporter, builder: Builder):  # pylint: disable=W0613

    import base64

    from aiq.plugins.opentelemetry.otlp_span_exporter import OTLPSpanExporter

    secret_key = config.secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = config.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not secret_key or not public_key:
        raise ValueError("secret and public keys are required for langfuse")

    credentials = f"{public_key}:{secret_key}".encode("utf-8")
    auth_header = base64.b64encode(credentials).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_header}"}

    yield OTLPSpanExporter(endpoint=config.endpoint, headers=headers)


class LangsmithTelemetryExporter(TelemetryExporterBaseConfig, name="langsmith"):
    """A telemetry exporter to transmit traces to externally hosted langsmith service."""

    endpoint: str = Field(
        description="The langsmith OTEL endpoint",
        default="https://api.smith.langchain.com/otel/v1/traces",
    )
    api_key: str = Field(description="The Langsmith API key", default="")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=LangsmithTelemetryExporter)
async def langsmith_telemetry_exporter(config: LangsmithTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Langsmith telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_exporter import OTLPSpanExporter

    api_key = config.api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("API key is required for langsmith")

    headers = {"x-api-key": api_key, "Langsmith-Project": config.project}
    yield OTLPSpanExporter(endpoint=config.endpoint, headers=headers)


class OtelCollectorTelemetryExporter(TelemetryExporterBaseConfig, name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    endpoint: str = Field(description="The otel endpoint to export telemetry traces.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create an OpenTelemetry telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_exporter import OTLPSpanExporter

    yield OTLPSpanExporter(endpoint=config.endpoint)


class PatronusTelemetryExporter(TelemetryExporterBaseConfig, name="patronus"):
    """A telemetry exporter to transmit traces to Patronus service."""

    endpoint: str = Field(description="The Patronus OTEL endpoint")
    api_key: str = Field(description="The Patronus API key", default="")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=PatronusTelemetryExporter)
async def patronus_telemetry_exporter(config: PatronusTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Patronus telemetry exporter."""

    from aiq.plugins.opentelemetry.otlp_span_exporter import OTLPSpanExporter

    api_key = config.api_key or os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        raise ValueError("API key is required for Patronus")

    headers = {
        "x-api-key": api_key,
        "pat-project-name": config.project,
    }
    yield OTLPSpanExporter(endpoint=config.endpoint, headers=headers)
