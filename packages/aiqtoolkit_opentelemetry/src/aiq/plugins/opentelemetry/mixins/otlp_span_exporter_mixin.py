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

    def __init__(self, *args, endpoint: str, headers: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self._exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)

    async def export(self, span: OtelSpan) -> None:
        """Export a batch of spans."""

        try:
            self._exporter.export([span])  # type: ignore
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
