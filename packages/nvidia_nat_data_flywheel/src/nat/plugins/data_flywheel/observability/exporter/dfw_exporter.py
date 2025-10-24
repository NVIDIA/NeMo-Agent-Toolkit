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
from abc import abstractmethod

from pydantic import BaseModel

from nat.builder.context import ContextState
from nat.data_models.span import Span
from nat.observability.exporter.span_exporter import SpanExporter
from nat.observability.processor.batching_processor import BatchingProcessor
from nat.observability.processor.falsy_batch_filter_processor import DictBatchFilterProcessor
from nat.observability.processor.processor_factory import processor_factory_from_type
from nat.observability.processor.processor_factory import processor_factory_to_type
from nat.plugins.data_flywheel.observability.processor import DFWToDictProcessor
from nat.plugins.data_flywheel.observability.processor import SpanToDFWRecordProcessor

logger = logging.getLogger(__name__)


class DictBatchingProcessor(BatchingProcessor[dict]):
    """Processor that batches dictionary objects for bulk operations.

    Specializes BatchingProcessor with explicit dict typing to support
    bulk export operations to sinks.
    """
    pass


class DFWExporter(SpanExporter[Span, dict]):
    """Abstract base class for Data Flywheel exporters."""

    def __init__(self,
                 export_contract: type[BaseModel],
                 context_state: ContextState | None = None,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 client_id: str = "default"):
        """Initialize the Data Flywheel exporter.

        Args:
            export_contract: The Pydantic model type for the export contract.
            context_state: The context state to use for the exporter.
            batch_size: The batch size for exporting spans.
            flush_interval: The flush interval in seconds for exporting spans.
            max_queue_size: The maximum queue size for exporting spans.
            drop_on_overflow: Whether to drop spans on overflow.
            shutdown_timeout: The shutdown timeout in seconds.
            client_id: The client ID for the exporter.
        """
        super().__init__(context_state)

        # Store the contract for property access
        self._export_contract = export_contract

        # Define the processor chain
        ConcreteSpanToDFWRecordProcessor = processor_factory_to_type(SpanToDFWRecordProcessor, export_contract)
        ConcreteDFWToDictProcessor = processor_factory_from_type(DFWToDictProcessor, export_contract)
        self.add_processor(ConcreteSpanToDFWRecordProcessor(client_id=client_id))  # type: ignore
        self.add_processor(ConcreteDFWToDictProcessor())
        self.add_processor(
            DictBatchingProcessor(batch_size=batch_size,
                                  flush_interval=flush_interval,
                                  max_queue_size=max_queue_size,
                                  drop_on_overflow=drop_on_overflow,
                                  shutdown_timeout=shutdown_timeout))
        self.add_processor(DictBatchFilterProcessor())

    @property
    def export_contract(self) -> type[BaseModel]:
        """The export contract used for processing spans before converting to dict.

        This type defines the structure of records that spans are converted to
        before being serialized to dictionaries for export.

        Returns:
            type[BaseModel]: The Pydantic model type for the export contract.
        """
        return self._export_contract

    @abstractmethod
    async def export_processed(self, item: dict | list[dict]) -> None:
        pass
