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

from aiq.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)


class OTLPSpanExporterMixin:
    """Mixin for OTLP span exporters.

    This mixin provides a default implementation of the export method for OTLP span exporters.
    It uses the OTLPSpanExporter from the opentelemetry.exporter.otlp.proto.http.trace_exporter module.

    Args:
        *args: Variable length argument list to pass to the superclass.
        endpoint (str): The endpoint of the OTLP service.
        headers (dict[str, str]): The headers to send with the request.
        **kwargs: Additional keyword arguments to pass to the superclass.

    Attributes:
        _exporter (OTLPSpanExporter): The OTLP span exporter.
    """

    def __init__(self, *args, endpoint: str, headers: dict[str, str], **kwargs):
        """Initialize the OTLP span exporter with the specified endpoint and headers."""
        super().__init__(*args, **kwargs)
        self._exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)

    async def export(self, span: OtelSpan) -> None:
        """Export an OtelSpan.

        Args:
            span (OtelSpan): The OtelSpan to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """

        try:
            self._exporter.export([span])  # type: ignore
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
