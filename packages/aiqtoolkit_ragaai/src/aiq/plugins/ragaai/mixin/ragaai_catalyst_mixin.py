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

import asyncio
import logging

import ragaai_catalyst
from ragaai_catalyst.tracers.exporters import DynamicTraceExporter

from aiq.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)


class RagaAICatalystMixin:
    """Mixin for RagaAI Catalyst exporters.

    This mixin provides RagaAI Catalyst-specific functionality for OpenTelemetry span exporters.
    It handles RagaAI Catalyst project and dataset configuration and uses the DynamicTraceExporter
    from the ragaai_catalyst.tracers.exporters module.

    Key Features:
    - RagaAI Catalyst authentication with access key and secret key
    - Project and dataset scoping for trace organization
    - Integration with RagaAI Catalyst's DynamicTraceExporter for telemetry transmission
    - Automatic initialization of RagaAI Catalyst client

    This mixin is designed to be used with OtelSpanExporter as a base class:

    Example:
        class MyCatalystExporter(OtelSpanExporter, RagaAICatalystMixin):
            def __init__(self, base_url, access_key, secret_key, project, dataset, **kwargs):
                super().__init__(base_url=base_url, access_key=access_key,
                               secret_key=secret_key, project=project, dataset=dataset, **kwargs)
    """

    def __init__(self, *args, base_url: str, access_key: str, secret_key: str, project: str, dataset: str, **kwargs):
        """Initialize the RagaAI Catalyst exporter.

        Args:
            base_url: RagaAI Catalyst base URL.
            access_key: RagaAI Catalyst access key.
            secret_key: RagaAI Catalyst secret key.
            project: RagaAI Catalyst project name.
            dataset: RagaAI Catalyst dataset name.
        """
        ragaai_catalyst.RagaAICatalyst(access_key=access_key, secret_key=secret_key, base_url=base_url)
        self._exporter = DynamicTraceExporter(project, dataset, base_url, "agentic/nemo-framework")
        super().__init__(*args, **kwargs)

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the RagaAI Catalyst exporter.

        Args:
            spans (list[OtelSpan]): The list of spans to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """
        try:
            # Run the blocking export operation in a thread pool to make it non-blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._exporter.export(spans))  # type: ignore[arg-type]
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
