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


@register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Phoenix telemetry exporter."""

    try:
        from aiq.plugins.phoenix.phoenix_exporter import PhoenixOtelExporter

        # Create the exporter
        yield PhoenixOtelExporter(endpoint=config.endpoint, project=config.project)

    except ConnectionError as ex:
        logger.warning("Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
                       ex,
                       exc_info=True)


# @register_telemetry_exporter(config_type=PhoenixTelemetryExporter)
# async def phoenix_telemetry_exporter(config: PhoenixTelemetryExporter, builder: Builder):  # pylint: disable=W0613
#     """Create a Phoenix telemetry exporter."""

#     try:
#         from aiq.plugins.phoenix.corrected_batched_exporter import BatchedPhoenixExporter

#         # Create the exporter
#         yield BatchedPhoenixExporter(endpoint=config.endpoint,
#                                      project=config.project,
#                                      batch_size=100,
#                                      flush_interval=5.0)

#     except ConnectionError as ex:
#         logger.warning("Unable to connect to Phoenix at port 6006. Are you sure Phoenix is running?\n %s",
#                        ex,
#                        exc_info=True)
