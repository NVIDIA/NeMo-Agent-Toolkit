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

from phoenix.otel import HTTPSpanExporter
from phoenix.otel import Resource
from phoenix.trace.projects import using_project

from aiq.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)


class PhoenixMixin:
    """Mixin for Phoenix exporters.

    This mixin provides a default implementation of the export method for Phoenix exporters.
    It uses the HTTPSpanExporter from the phoenix.otel module.

    Args:
        *args: Variable length argument list to pass to the superclass.
        endpoint (str): The endpoint of the Phoenix service.
        project (str): The project name to group the telemetry traces.
        **kwargs: Additional keyword arguments to pass to the superclass.

    Attributes:
        _exporter (HTTPSpanExporter): The Phoenix span exporter.
        _project (str): The project name to group the telemetry traces.
        _resource (Resource): The resource to add to the span.
    """

    def __init__(self, *args, endpoint: str, project: str, **kwargs):
        """Initialize the Phoenix exporter with the specified endpoint and project."""
        super().__init__(*args, **kwargs)
        self._exporter = HTTPSpanExporter(endpoint=endpoint)
        self._project = project
        self._resource = Resource(attributes={
            "openinference.project.name": project,
        })

    async def export(self, span: OtelSpan) -> None:
        """Export an OtelSpan.

        Args:
            span (OtelSpan): The OtelSpan to export.
        """

        try:
            with using_project(self._project):
                span.set_resource(self._resource)
                self._exporter.export([span])  # type: ignore
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
