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

from aiq.builder.context import AIQContextState
from aiq.plugins.opentelemetry.mixin.otlp_span_exporter_mixin import OTLPSpanExporterMixin
from aiq.plugins.opentelemetry.otel_span_exporter import OtelSpanExporter

logger = logging.getLogger(__name__)


class OTLPSpanAdapterExporter(OTLPSpanExporterMixin, OtelSpanExporter):  # pylint: disable=R0901
    """An OpenTelemetry OTLP span exporter for sending traces to OTLP-compatible services.

    This class combines the OtelSpanExporter base functionality with OTLP-specific
    export capabilities to provide a complete solution for sending telemetry traces
    to any OTLP-compatible collector or service via HTTP.

    Key Features:
    - Complete span processing pipeline (IntermediateStep → Span → OtelSpan → Export)
    - Batching support for efficient transmission
    - OTLP HTTP protocol for maximum compatibility
    - Configurable authentication via headers
    - Resource attribute management
    - Error handling and retry logic

    This exporter is commonly used with services like:
    - OpenTelemetry Collector
    - Jaeger (OTLP endpoint)
    - Grafana Tempo
    - Custom OTLP-compatible backends

    Example:
        exporter = OTLPSpanAdapterExporter(
            endpoint="https://api.service.com/v1/traces",
            headers={"Authorization": "Bearer your-token"},
            batch_size=50,
            flush_interval=10.0
        )
    """

    def __init__(self,
                 context_state: AIQContextState | None = None,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 **otlp_kwargs):
        """Initialize the OTLP span exporter.

        Args:
            context_state: The context state for the exporter.
            batch_size: Number of spans to batch before exporting.
            flush_interval: Time in seconds between automatic batch flushes.
            max_queue_size: Maximum number of spans to queue.
            drop_on_overflow: Whether to drop spans when queue is full.
            shutdown_timeout: Maximum time to wait for export completion during shutdown.
        """
        super().__init__(context_state=context_state,
                         batch_size=batch_size,
                         flush_interval=flush_interval,
                         max_queue_size=max_queue_size,
                         drop_on_overflow=drop_on_overflow,
                         shutdown_timeout=shutdown_timeout,
                         **otlp_kwargs)
