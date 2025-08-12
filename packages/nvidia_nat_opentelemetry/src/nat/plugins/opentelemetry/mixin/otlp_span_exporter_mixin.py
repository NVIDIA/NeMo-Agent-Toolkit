# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from nat.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)


class OTLPSpanExporterMixin:
    """Mixin for OTLP span exporters.

    This mixin provides OTLP-specific functionality for OpenTelemetry span exporters.
    It handles OTLP protocol transmission using the standard OpenTelemetry OTLP HTTP exporter.

    Key Features:
    - Standard OTLP HTTP protocol support for span export
    - Configurable endpoint and headers for authentication/routing
    - Integration with OpenTelemetry's OTLPSpanExporter for reliable transmission
    - Works with any OTLP-compatible collector or service

    This mixin is designed to be used with OtelSpanExporter as a base class:

    Example:
        class MyOTLPExporter(OtelSpanExporter, OTLPSpanExporterMixin):
            def __init__(self, endpoint, headers, **kwargs):
                super().__init__(endpoint=endpoint, headers=headers, **kwargs)
    """

    def __init__(self, *args, endpoint: str, headers: dict[str, str] | None = None, **kwargs):
        """Initialize the OTLP span exporter.

        Args:
            endpoint: OTLP service endpoint URL.
            headers: HTTP headers for authentication and metadata.
        """
        # Initialize exporter before super().__init__() to ensure it's available
        # if parent class initialization potentially calls export_otel_spans()
        self._exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        super().__init__(*args, **kwargs)

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the OTLP exporter.

        Args:
            spans (list[OtelSpan]): The list of spans to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """
        try:
            self._exporter.export(spans)  # type: ignore[arg-type]
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
