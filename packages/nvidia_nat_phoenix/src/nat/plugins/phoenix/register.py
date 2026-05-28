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

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin

logger = logging.getLogger(__name__)


def _phoenix_auth_headers(api_key: str) -> dict[str, str]:
    """Build Phoenix OTLP auth headers."""
    if api_key.lower().startswith("bearer "):
        bearer_token = api_key
    else:
        bearer_token = f"Bearer {api_key}"

    return {"authorization": bearer_token}


class PhoenixTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="phoenix"):
    """A telemetry exporter to transmit traces to externally hosted phoenix service."""

    endpoint: str = Field(
        description="Phoenix server endpoint for trace export (e.g., 'http://localhost:6006/v1/traces')")
    timeout: float = Field(default=30.0, description="Timeout in seconds for HTTP requests to Phoenix server")
    api_key: SerializableSecretStr = Field(
        description="Phoenix API key. If empty, uses the PHOENIX_API_KEY environment variable.",
        default_factory=lambda: SerializableSecretStr(""),
    )


@register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder):
    """Create a Phoenix telemetry exporter."""

    try:
        from nat.plugins.phoenix.phoenix_exporter import PhoenixOtelExporter

        api_key = get_secret_value(config.api_key) if config.api_key else None
        api_key = (api_key or os.environ.get("PHOENIX_API_KEY") or "").strip()
        headers = _phoenix_auth_headers(api_key) if api_key else None

        # Create the exporter
        yield PhoenixOtelExporter(endpoint=config.endpoint,
                                  project=config.project,
                                  timeout=config.timeout,
                                  headers=headers,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)

    except ConnectionError as ex:
        logger.warning("Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
                       ex,
                       exc_info=True)
