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

from aiq.builder.context import AIQContextState
from aiq.data_models.span import Span
from aiq.observability.exporter.span_exporter import SpanExporter
from aiq.observability.processor.batching_processor import BatchingProcessor
from aiq.observability.processor.processor import Processor
from aiq.plugins.opentelemetry.otel_span import OtelSpan
from aiq.plugins.opentelemetry.span_converter import convert_span_to_otel

logger = logging.getLogger(__name__)


class SpanToOtelProcessor(Processor[Span, OtelSpan]):
    """Processor that converts a Span to an OtelSpan."""

    async def process(self, item: Span) -> OtelSpan:
        return convert_span_to_otel(item)  # type: ignore


class OtelSpanBatchProcessor(BatchingProcessor[OtelSpan]):
    """Processor that batches OtelSpans with explicit type information.

    This class provides explicit type information for the TypeIntrospectionMixin
    by overriding the type properties directly.
    """
    pass


class OtelSpanExporter(SpanExporter[Span, OtelSpan]):
    """Abstract base class for OpenTelemetry exporters.

    This class provides a specialized implementation for OpenTelemetry exporters.
    It builds upon SpanExporter's span construction logic and automatically adds
    a SpanToOtelProcessor to transform Span objects into OtelSpan objects.

    The processing flow is:
    IntermediateStep → Span → OtelSpan → Export

    Key Features:
    - Automatic span construction from IntermediateStep events (via SpanExporter)
    - Built-in Span to OtelSpan conversion (via SpanToOtelProcessor)
    - Support for additional processing steps if needed
    - Type-safe processing pipeline with enhanced TypeVar compatibility
    - Batching support for efficient export

    Inheritance Hierarchy:
    - BaseExporter: Core functionality + TypeIntrospectionMixin
    - ProcessingExporter: Processor pipeline support
    - SpanExporter: Span creation and lifecycle management
    - OtelExporter: OpenTelemetry-specific span transformation

    Generic Types:
    - InputSpanT: Always Span (from IntermediateStep conversion)
    - OutputSpanT: Always OtelSpan (for OpenTelemetry compatibility)

    Args:
        context_state (AIQContextState | None): The context state to use for the exporter.
        batch_size (int): The batch size for exporting.
        flush_interval (float): The flush interval for exporting.
        max_queue_size (int): The maximum queue size for exporting.
        drop_on_overflow (bool): Whether to drop on overflow for exporting.
        shutdown_timeout (float): The shutdown timeout for exporting.
    """

    def __init__(self,
                 context_state: AIQContextState | None = None,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0):
        """Initialize the OpenTelemetry exporter with the specified context state."""
        super().__init__(context_state)

        self._batching_processor = OtelSpanBatchProcessor(batch_size=batch_size,
                                                          flush_interval=flush_interval,
                                                          max_queue_size=max_queue_size,
                                                          drop_on_overflow=drop_on_overflow,
                                                          shutdown_timeout=shutdown_timeout,
                                                          done_callback=self.export_processed)

        self.add_processor(SpanToOtelProcessor())
        self.add_processor(self._batching_processor)

    @abstractmethod
    async def export_processed(self, item: OtelSpan | list[OtelSpan]) -> None:
        """Export the processed span(s).

        Args:
            item (OtelSpan | list[OtelSpan]): The processed span(s) to export.
                Can be a single span or a batch of spans from BatchingProcessor.
        """
        pass
