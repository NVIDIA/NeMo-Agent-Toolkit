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

from openinference.instrumentation import dangerously_using_project

from nat.plugins.opentelemetry.otel_span import OtelSpan
from phoenix.otel import HTTPSpanExporter

logger = logging.getLogger(__name__)


class PhoenixMixin:
    """Mixin for Phoenix exporters.

    This mixin provides Phoenix-specific functionality for OpenTelemetry span exporters.
    It handles Phoenix project scoping and uses the HTTPSpanExporter from the phoenix.otel module.

    Key Features:
    - Automatic Phoenix project name injection into resource attributes
    - Phoenix project scoping via using_project() context manager
    - Integration with Phoenix's HTTPSpanExporter for telemetry transmission

    This mixin is designed to be used with OtelSpanExporter as a base class:

    Example::

        class MyPhoenixExporter(OtelSpanExporter, PhoenixMixin):
            def __init__(self, endpoint, project, **kwargs):
                super().__init__(endpoint=endpoint, project=project, **kwargs)
    """

    def __init__(self, *args, endpoint: str, project: str, **kwargs):
        """Initialize the Phoenix exporter.

        Args:
            endpoint: Phoenix service endpoint URL.
            project: Phoenix project name for trace grouping.
        """
        self._exporter = HTTPSpanExporter(endpoint=endpoint)
        self._project = project

        # Add Phoenix project name to resource attributes
        kwargs.setdefault('resource_attributes', {})
        kwargs['resource_attributes'].update({'openinference.project.name': project})

        super().__init__(*args, **kwargs)

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the Phoenix exporter.

        Args:
            spans (list[OtelSpan]): The list of spans to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """
        try:
            with dangerously_using_project(self._project):
                self._exporter.export(spans)  # type: ignore
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
