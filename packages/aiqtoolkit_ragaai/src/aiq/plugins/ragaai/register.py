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


class CatalystTelemetryExporter(TelemetryExporterBaseConfig, name="catalyst"):
    """A telemetry exporter to transmit traces to RagaAI catalyst."""
    endpoint: str = Field(description="The RagaAI Catalyst endpoint", default="")
    access_key: str = Field(description="The RagaAI Catalyst API access key", default="")
    secret_key: str = Field(description="The RagaAI Catalyst API secret key", default="")
    project: str = Field(description="The RagaAI Catalyst project name")
    dataset: str | None = Field(description="The RagaAI Catalyst dataset name", default=None)

    # Local file control options
    disable_local_file: bool = Field(description="Disable creation of local rag_agent_traces.json file", default=True)
    local_file_path: str | None = Field(
        description="Custom path to save local trace files instead of current directory", default=None)


@register_telemetry_exporter(config_type=CatalystTelemetryExporter)
async def catalyst_telemetry_exporter(config: CatalystTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Catalyst telemetry exporter."""

    try:
        import os

        from aiq.plugins.ragaai.ragaai_catalyst_exporter import RagaAICatalystExporter

        access_key = config.access_key or os.environ.get("CATALYST_ACCESS_KEY")
        secret_key = config.secret_key or os.environ.get("CATALYST_SECRET_KEY")
        endpoint = config.endpoint or os.environ.get("CATALYST_ENDPOINT")
        project = config.project
        dataset = config.dataset

        assert endpoint is not None, "catalyst endpoint is not set"
        assert access_key is not None, "catalyst access key is not set"
        assert secret_key is not None, "catalyst secret key is not set"

        yield RagaAICatalystExporter(base_url=endpoint,
                                     access_key=access_key,
                                     secret_key=secret_key,
                                     project=project,
                                     dataset=dataset,
                                     disable_local_file=config.disable_local_file,
                                     local_file_path=config.local_file_path)
    except Exception as e:
        logger.warning("Error creating catalyst telemetry exporter: %s", e, exc_info=True)
