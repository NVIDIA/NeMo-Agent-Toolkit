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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class PhoenixTelemetryExporter(TelemetryExporterBaseConfig, name="phoenix"):
    """A telemetry exporter to transmit traces to externally hosted phoenix service."""

    endpoint: str = Field(
        description="Phoenix server endpoint for trace export (e.g., 'http://localhost:6006/v1/traces'")
    project: str = Field(description="The project name to group the telemetry traces.")
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")


@register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Phoenix telemetry exporter."""

    try:
        from aiq.plugins.phoenix.phoenix_exporter import PhoenixOtelExporter

        # Create the exporter
        yield PhoenixOtelExporter(endpoint=config.endpoint,
                                  project=config.project,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)

    except ConnectionError as ex:
        logger.warning("Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
                       ex,
                       exc_info=True)
