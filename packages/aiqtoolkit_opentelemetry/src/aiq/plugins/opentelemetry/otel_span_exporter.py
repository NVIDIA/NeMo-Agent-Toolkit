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
from typing import TypeVar

from aiq.builder.context import AIQContextState
from aiq.data_models.span import Span
from aiq.observability.exporter.span_exporter import SpanExporter
from aiq.observability.processor.processor import Processor
from aiq.plugins.opentelemetry.otel_span import OtelSpan
from aiq.plugins.opentelemetry.span_converter import convert_span_to_otel

logger = logging.getLogger(__name__)

SpanT = TypeVar("SpanT", bound=Span)
OtelSpanT = TypeVar("OtelSpanT", bound=OtelSpan)


class SpanToOtelProcessor(Processor[Span, OtelSpan]):
    """Processor that converts a Span to an OtelSpan."""

    async def process(self, item: Span) -> OtelSpan:
        return convert_span_to_otel(item)


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
    - Type-safe processing pipeline

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
    """

    def __init__(self, context_state: AIQContextState | None = None):
        """Initialize the OpenTelemetry exporter with the specified context state."""
        super().__init__(context_state)
        self.add_processor(SpanToOtelProcessor())

    @abstractmethod
    async def export_processed(self, item: OtelSpan) -> None:
        """Export the processed span.

        Args:
            item (OtelSpan): The processed span to export.
        """
        pass
