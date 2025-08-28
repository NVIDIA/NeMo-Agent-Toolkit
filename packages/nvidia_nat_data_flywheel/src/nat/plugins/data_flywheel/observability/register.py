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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch import ContractVersion

logger = logging.getLogger(__name__)


class DFWElasticsearchTelemetryExporter(TelemetryExporterBaseConfig,
                                        BatchConfigMixin,
                                        name="data_flywheel_elasticsearch"):
    """A telemetry exporter to transmit traces to NVIDIA Data Flywheel via Elasticsearch."""

    client_id: str = Field(description="The data flywheel client ID.")
    index: str = Field(description="The elasticsearch index name.")
    endpoint: str = Field(description="The elasticsearch endpoint.")
    contract_version: ContractVersion = Field(default=ContractVersion.V1_1,
                                              description="The DFW Elasticsearch record schema version to use.")
    username: str | None = Field(default=None, description="The elasticsearch username.")
    password: str | None = Field(default=None, description="The elasticsearch password.")
    headers: dict | None = Field(default=None, description="Additional headers for elasticsearch requests.")


@register_telemetry_exporter(config_type=DFWElasticsearchTelemetryExporter)
async def dfw_elasticsearch_telemetry_exporter(config: DFWElasticsearchTelemetryExporter, _builder: Builder):
    # pylint: disable=import-outside-toplevel
    from nat.plugins.data_flywheel.observability.exporter.dfw_elasticsearch_exporter import DFWElasticsearchExporter

    elasticsearch_auth = (config.username, config.password) if config.username and config.password else ()

    yield DFWElasticsearchExporter(client_id=config.client_id,
                                   index=config.index,
                                   endpoint=config.endpoint,
                                   elasticsearch_auth=elasticsearch_auth,
                                   headers=config.headers,
                                   contract_version=config.contract_version,
                                   batch_size=config.batch_size,
                                   flush_interval=config.flush_interval,
                                   max_queue_size=config.max_queue_size,
                                   drop_on_overflow=config.drop_on_overflow,
                                   shutdown_timeout=config.shutdown_timeout)
