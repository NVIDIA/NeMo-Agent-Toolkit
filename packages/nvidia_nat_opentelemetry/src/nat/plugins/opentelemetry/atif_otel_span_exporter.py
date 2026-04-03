# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""OpenTelemetry variant of the ATIF Trajectory span exporter.

Mirrors OtelSpanExporter but uses ATIFTrajectorySpanExporter as the base,
converting ATIFTrajectory → Span → OtelSpan for export.
"""

import logging
from abc import abstractmethod

from nat.builder.context import ContextState
from nat.observability.exporter.atif_trajectory_span_exporter import ATIFTrajectorySpanExporter
from nat.plugins.opentelemetry.otel_span import OtelSpan
from nat.plugins.opentelemetry.otel_span_exporter import OtelSpanBatchProcessor
from nat.plugins.opentelemetry.otel_span_exporter import SpanToOtelProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


class ATIFOtelSpanExporter(ATIFTrajectorySpanExporter[OtelSpan]):
    """ATIF-based OpenTelemetry span exporter.

    Processing flow:
        IntermediateStep (collected) → ATIFTrajectory → Span → OtelSpan → Export

    Mirrors OtelSpanExporter's pipeline (SpanToOtelProcessor + OtelSpanBatchProcessor)
    but uses ATIFTrajectorySpanExporter for span generation from ATIF data.
    """

    def __init__(
        self,
        context_state: ContextState | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
        drop_on_overflow: bool = False,
        shutdown_timeout: float = 10.0,
        resource_attributes: dict[str, str] | None = None,
    ):
        super().__init__(context_state=context_state)

        if resource_attributes is None:
            resource_attributes = {}
        self._resource = Resource(attributes=resource_attributes)

        self.add_processor(SpanToOtelProcessor())
        self.add_processor(
            OtelSpanBatchProcessor(
                batch_size=batch_size,
                flush_interval=flush_interval,
                max_queue_size=max_queue_size,
                drop_on_overflow=drop_on_overflow,
                shutdown_timeout=shutdown_timeout,
            ))

    async def export_processed(self, item: OtelSpan | list[OtelSpan]) -> None:
        """Export the processed span(s) with resource attributes."""
        try:
            if isinstance(item, OtelSpan):
                spans = [item]
            elif isinstance(item, list):
                spans = item
            else:
                logger.warning("Unexpected item type: %s", type(item))
                return

            for span in spans:
                span.set_resource(self._resource)

            await self.export_otel_spans(spans)

        except Exception as e:
            logger.error("Error exporting ATIF-based spans: %s", e, exc_info=True)

    @abstractmethod
    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OpenTelemetry spans.

        Must be implemented by concrete exporters (Phoenix, OTLP, etc.).
        """
        pass
