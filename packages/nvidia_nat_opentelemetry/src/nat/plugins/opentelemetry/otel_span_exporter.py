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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from nat.builder.context import ContextState
from nat.data_models.span import Span
from nat.observability.exporter.span_exporter import SpanExporter
from nat.observability.processor.batching_processor import BatchingProcessor
from nat.observability.processor.processor import Processor
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.span_converter import convert_span_to_otel
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


def get_opentelemetry_sdk_version() -> str:
    """Get the OpenTelemetry SDK version dynamically.

    Returns:
        The version of the opentelemetry-sdk package, or 'unknown' if not found.
    """
    try:
        return version("opentelemetry-sdk")
    except PackageNotFoundError:
        logger.warning("Could not determine opentelemetry-sdk version")
        return "unknown"


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
    """

    def __init__(self,
                 context_state: ContextState | None = None,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 resource_attributes: dict[str, str] | None = None):
        """Initialize the OpenTelemetry exporter.

        Args:
            context_state: The context state to use for the exporter.
            batch_size: The batch size for exporting spans.
            flush_interval: The flush interval in seconds for exporting spans.
            max_queue_size: The maximum queue size for exporting spans.
            drop_on_overflow: Whether to drop spans on overflow.
            shutdown_timeout: The shutdown timeout in seconds.
            resource_attributes: Additional resource attributes for spans.
        """
        super().__init__(context_state)

        # Initialize resource for span attribution
        if resource_attributes is None:
            resource_attributes = {}
        self._resource = Resource(attributes=resource_attributes)

        self.add_processor(SpanToOtelProcessor())
        self.add_processor(
            OtelSpanBatchProcessor(batch_size=batch_size,
                                   flush_interval=flush_interval,
                                   max_queue_size=max_queue_size,
                                   drop_on_overflow=drop_on_overflow,
                                   shutdown_timeout=shutdown_timeout))

    async def export_processed(self, item: OtelSpan | list[OtelSpan]) -> None:
        """Export the processed span(s).

        This method handles the common logic for all OTEL exporters:
        - Normalizes single spans vs. batches
        - Sets resource attributes on spans
        - Delegates to the abstract export_otel_spans method

        Args:
            item (OtelSpan | list[OtelSpan]): The processed span(s) to export.
                Can be a single span or a batch of spans from BatchingProcessor.
        """
        try:
            if isinstance(item, OtelSpan):
                spans = [item]
            elif isinstance(item, list):
                spans = item
            else:
                logger.warning("Unexpected item type: %s", type(item))
                return

            # Set resource attributes on all spans
            for span in spans:
                span.set_resource(self._resource)

            # Delegate to concrete implementation
            await self.export_otel_spans(spans)

        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)

    @abstractmethod
    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OpenTelemetry spans.

        This method must be implemented by concrete exporters to handle
        the actual export logic (e.g., HTTP requests, file writes, etc.).

        Args:
            spans (list[OtelSpan]): The list of spans to export.
        """
        pass
