# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin

logger = logging.getLogger(__name__)

# Arize AX OTLP defaults (align with arize-otel `Endpoint` and HTTPS trace path)
_ARIZE_US_OTLP_GRPC = "https://otlp.arize.com/v1"
_ARIZE_US_OTLP_HTTP = "https://otlp.arize.com/v1/traces"
_ARIZE_EU_OTLP_GRPC = "https://otlp.eu-west-1a.arize.com/v1"
_ARIZE_EU_OTLP_HTTP = "https://otlp.eu-west-1a.arize.com/v1/traces"


def _arize_ax_auth_headers(*, space_id: str, api_key: str) -> dict[str, str]:
    """Build OTLP metadata headers for Arize AX (same keys as arize-otel `arize.otel.otel._get_arize_auth_headers`)."""
    return {
        "authorization": api_key,
        "api_key": api_key,
        "arize-space-id": space_id,
        "space_id": space_id,
        "arize-interface": "otel",
    }


def _arize_ax_default_endpoint(*, protocol: Literal['http', 'grpc'], use_eu_region: bool) -> str:
    """Return the default Arize AX OTLP collector URL for the given protocol and region."""
    if use_eu_region:
        return _ARIZE_EU_OTLP_HTTP if protocol == "http" else _ARIZE_EU_OTLP_GRPC
    return _ARIZE_US_OTLP_HTTP if protocol == "http" else _ARIZE_US_OTLP_GRPC


class LangfuseTelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="langfuse"):
    """A telemetry exporter to transmit traces to externally hosted langfuse service."""

    endpoint: str = Field(description="The langfuse OTEL endpoint (/api/public/otel/v1/traces)")
    public_key: SerializableSecretStr = Field(description="The Langfuse public key",
                                              default_factory=lambda: SerializableSecretStr(""))
    secret_key: SerializableSecretStr = Field(description="The Langfuse secret key",
                                              default_factory=lambda: SerializableSecretStr(""))
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=LangfuseTelemetryExporter)
async def langfuse_telemetry_exporter(config: LangfuseTelemetryExporter, builder: Builder):

    import base64

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    secret_key = get_secret_value(config.secret_key) if config.secret_key else os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = get_secret_value(config.public_key) if config.public_key else os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not secret_key or not public_key:
        raise ValueError("secret and public keys are required for langfuse")

    credentials = f"{public_key}:{secret_key}".encode()
    auth_header = base64.b64encode(credentials).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_header}"}

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )


class LangsmithTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="langsmith"):
    """A telemetry exporter to transmit traces to externally hosted langsmith service."""

    endpoint: str = Field(
        description="The langsmith OTEL endpoint",
        default="https://api.smith.langchain.com/otel/v1/traces",
    )
    api_key: SerializableSecretStr = Field(description="The Langsmith API key",
                                           default_factory=lambda: SerializableSecretStr(""))
    workspace_id: str = Field(
        default="", description="The Langsmith workspace ID. Falls back to LANGSMITH_WORKSPACE_ID env var if not set.")
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=LangsmithTelemetryExporter)
async def langsmith_telemetry_exporter(config: LangsmithTelemetryExporter, builder: Builder):
    """Create a Langsmith telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    api_key = get_secret_value(config.api_key) if config.api_key else os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("API key is required for langsmith")

    headers = {"x-api-key": api_key, "Langsmith-Project": config.project}
    workspace_id = (config.workspace_id or os.environ.get("LANGSMITH_WORKSPACE_ID")
                    or os.environ.get("LANGCHAIN_WORKSPACE_ID"))
    if workspace_id:
        headers["X-Tenant-Id"] = workspace_id
    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )


class OtelCollectorTelemetryExporter(BatchConfigMixin,
                                     CollectorConfigMixin,
                                     TelemetryExporterBaseConfig,
                                     name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder):
    """Create an OpenTelemetry telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
    from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version

    # Default resource attributes
    default_resource_attributes = {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": get_opentelemetry_sdk_version(),
        "service.name": config.project,
    }

    # Merge defaults with config, giving precedence to config
    merged_resource_attributes = {**default_resource_attributes, **config.resource_attributes}

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        resource_attributes=merged_resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )


class PatronusTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="patronus"):
    """A telemetry exporter to transmit traces to Patronus service."""

    api_key: SerializableSecretStr = Field(description="The Patronus API key",
                                           default_factory=lambda: SerializableSecretStr(""))
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=PatronusTelemetryExporter)
async def patronus_telemetry_exporter(config: PatronusTelemetryExporter, builder: Builder):
    """Create a Patronus telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    api_key = get_secret_value(config.api_key) if config.api_key else os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        raise ValueError("API key is required for Patronus")

    headers = {
        "x-api-key": api_key,
        "pat-project-name": config.project,
    }
    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
        protocol="grpc",
    )


class GalileoTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="galileo"):
    """A telemetry exporter to transmit traces to externally hosted galileo service."""

    endpoint: str = Field(
        description="The galileo endpoint to export telemetry traces.",
        default="https://app.galileo.ai/api/galileo/otel/traces",
    )
    logstream: str = Field(description="The logstream name to group the telemetry traces.")
    api_key: SerializableSecretStr = Field(description="The api key to authenticate with the galileo service.")


@register_telemetry_exporter(config_type=GalileoTelemetryExporter)
async def galileo_telemetry_exporter(config: GalileoTelemetryExporter, builder: Builder):
    """Create a Galileo telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    headers = {
        "Galileo-API-Key": get_secret_value(config.api_key),
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


class WeaveOtelTelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="weave_otel"):
    """A telemetry exporter to transmit traces to Weights & Biases Weave via OTel."""

    endpoint: str = Field(
        description="The W&B Weave OTel endpoint",
        default="https://trace.wandb.ai/otel/v1/traces",
    )
    api_key: SerializableSecretStr = Field(description="The W&B API key",
                                           default_factory=lambda: SerializableSecretStr(""))
    project: str = Field(description="The W&B project name.")
    entity: str = Field(description="The W&B username or team name.")


@register_telemetry_exporter(config_type=WeaveOtelTelemetryExporter)
async def weave_otel_telemetry_exporter(config: WeaveOtelTelemetryExporter, builder: Builder):
    """Create a Weave OTel telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    api_key = get_secret_value(config.api_key) if config.api_key else os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("API key is required for Weave (set api_key or WANDB_API_KEY env var)")

    headers = {"wandb-api-key": api_key}
    resource_attributes = {
        "wandb.project": config.project,
        "wandb.entity": config.entity,
    }

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        resource_attributes=resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )

    
class ArizeAxTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="arize_ax"):
    """Export traces to Arize AX over OTLP.

    See Arize AX OpenTelemetry docs. Headers match ``arize-otel`` (``authorization``, ``arize-space-id``, etc.)
    with default US or EU collectors when ``endpoint`` is unset.
    """

    project: str = Field(
        default="",
        description="Arize project name. If empty, uses the ARIZE_PROJECT_NAME environment variable.",
    )
    space_id: str = Field(
        default="",
        description="Arize space ID. If empty, uses the ARIZE_SPACE_ID environment variable.",
    )
    api_key: SerializableSecretStr = Field(
        description="Arize API key. If empty, uses the ARIZE_API_KEY environment variable.",
        default_factory=lambda: SerializableSecretStr(""),
    )
    endpoint: str | None = Field(
        default=None,
        description="OTLP URL. If unset, uses the default US or EU host for the chosen protocol (HTTP or gRPC).",
    )
    use_eu_region: bool = Field(
        default=False,
        description="When true and endpoint is unset, use the Arize EU OTLP collector (otlp.eu-west-1a.arize.com).",
    )
    protocol: Literal['http', 'grpc'] = Field(
        default="grpc",
        description="OTLP transport. HTTP uses the /v1/traces path; gRPC uses the /v1 host path per Arize defaults.",
    )
    resource_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Extra OpenTelemetry resource attributes merged with the project name.",
    )


@register_telemetry_exporter(config_type=ArizeAxTelemetryExporter)
async def arize_ax_telemetry_exporter(config: ArizeAxTelemetryExporter, builder: Builder):
    """Create a telemetry exporter that sends OTLP traces to Arize AX."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    space_id = (config.space_id or os.environ.get("ARIZE_SPACE_ID") or "").strip()
    if not space_id:
        raise ValueError("space_id is required for Arize AX (set in config or ARIZE_SPACE_ID)")

    api_key = get_secret_value(config.api_key) if config.api_key else None
    api_key = api_key or os.environ.get("ARIZE_API_KEY")
    if not api_key:
        raise ValueError("api_key is required for Arize AX (set in config or ARIZE_API_KEY)")

    project_name = (config.project or os.environ.get("ARIZE_PROJECT_NAME") or "").strip()
    if not project_name:
        raise ValueError("project is required for Arize AX (set `project` on the exporter or ARIZE_PROJECT_NAME)")

    endpoint = config.endpoint or _arize_ax_default_endpoint(
        protocol=config.protocol,
        use_eu_region=config.use_eu_region,
    )
    headers = _arize_ax_auth_headers(space_id=space_id, api_key=api_key)

    default_resource_attributes = {
        "openinference.project.name": project_name,
    }
    merged_resource_attributes = {**default_resource_attributes, **config.resource_attributes}

    yield OTLPSpanAdapterExporter(
        endpoint=endpoint,
        headers=headers,
        protocol=config.protocol,
        resource_attributes=merged_resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )
