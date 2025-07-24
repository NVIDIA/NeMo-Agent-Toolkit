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
from aiq.observability.mixin.batch_config_mixin import BatchConfigMixin
from aiq.observability.mixin.collector_config_mixin import CollectorConfigMixin

logger = logging.getLogger(__name__)


class CatalystTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="catalyst"):
    """A telemetry exporter to transmit traces to RagaAI catalyst."""
    endpoint: str = Field(description="The RagaAI Catalyst endpoint", default="https://catalyst.raga.ai/api")
    access_key: str = Field(description="The RagaAI Catalyst API access key", default="")
    secret_key: str = Field(description="The RagaAI Catalyst API secret key", default="")
    dataset: str | None = Field(description="The RagaAI Catalyst dataset name", default=None)
    tracer_type: str = Field(description="The RagaAI Catalyst tracer type", default="agentic/nemo-framework")

    # Debug mode control options
    debug_mode: bool = Field(description="When False (default), creates local rag_agent_traces.json file. "
                             "When True, skips local file creation for cleaner operation.",
                             default=False)


@register_telemetry_exporter(config_type=CatalystTelemetryExporter)
async def catalyst_telemetry_exporter(config: CatalystTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """Create a Catalyst telemetry exporter."""

    try:
        import os

        from aiq.plugins.ragaai.ragaai_catalyst_exporter import RagaAICatalystExporter

        access_key = config.access_key or os.environ.get("CATALYST_ACCESS_KEY")
        secret_key = config.secret_key or os.environ.get("CATALYST_SECRET_KEY")
        endpoint = config.endpoint or os.environ.get("CATALYST_ENDPOINT")

        assert endpoint is not None, "catalyst endpoint is not set"
        assert access_key is not None, "catalyst access key is not set"
        assert secret_key is not None, "catalyst secret key is not set"

        yield RagaAICatalystExporter(base_url=endpoint,
                                     access_key=access_key,
                                     secret_key=secret_key,
                                     project=config.project,
                                     dataset=config.dataset,
                                     tracer_type=config.tracer_type,
                                     debug_mode=config.debug_mode,
                                     batch_size=config.batch_size,
                                     flush_interval=config.flush_interval,
                                     max_queue_size=config.max_queue_size,
                                     drop_on_overflow=config.drop_on_overflow,
                                     shutdown_timeout=config.shutdown_timeout)
    except Exception as e:
        logger.warning("Error creating catalyst telemetry exporter: %s", e, exc_info=True)
