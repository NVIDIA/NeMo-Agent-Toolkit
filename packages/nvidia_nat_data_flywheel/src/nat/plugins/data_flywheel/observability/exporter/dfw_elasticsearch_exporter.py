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

from pydantic import BaseModel

from nat.builder.context import AIQContextState
from nat.plugins.data_flywheel.observability.exporter.dfw_exporter import DFWExporter
from nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin import ElasticsearchMixin
from nat.plugins.data_flywheel.observability.schema.contract_version import ElasticsearchContractVersion

logger = logging.getLogger(__name__)


class DFWElasticsearchExporter(ElasticsearchMixin, DFWExporter):
    """Elasticsearch-specific Data Flywheel exporter."""

    def __init__(self,
                 context_state: AIQContextState | None = None,
                 client_id: str = "default",
                 contract_version: ElasticsearchContractVersion = ElasticsearchContractVersion.VERSION_1_1,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 **elasticsearch_kwargs):
        """Initialize the Elasticsearch Data Flywheel exporter.

        Args:
            context_state: The context state to use for the exporter.
            client_id: The client ID for the exporter.
            contract_version: The contract version to use for the exporter.
            batch_size: The batch size for exporting spans.
            flush_interval: The flush interval in seconds for exporting spans.
            max_queue_size: The maximum queue size for exporting spans.
            drop_on_overflow: Whether to drop spans on overflow.
            shutdown_timeout: The shutdown timeout in seconds.
            **elasticsearch_kwargs: Additional arguments for ElasticsearchMixin:
                - endpoint: The elasticsearch endpoint.
                - index: The elasticsearch index name.
                - elasticsearch_auth: The elasticsearch authentication credentials.
                - headers: The elasticsearch headers.
        """
        # Initialize both mixins - ElasticsearchMixin expects elasticsearch_kwargs,
        # DFWExporter expects the standard exporter parameters
        self.contract_version = contract_version
        super().__init__(context_state=context_state,
                         batch_size=batch_size,
                         flush_interval=flush_interval,
                         max_queue_size=max_queue_size,
                         drop_on_overflow=drop_on_overflow,
                         shutdown_timeout=shutdown_timeout,
                         client_id=client_id,
                         **elasticsearch_kwargs)

    @property
    def export_contract(self) -> type[BaseModel]:
        """The export contract for Elasticsearch.
        """
        return self.contract_version.value

    async def export_processed(self, item: dict | list[dict]) -> None:
        """Export processed DFW records to Elasticsearch."""
        await super().export_processed(item)
