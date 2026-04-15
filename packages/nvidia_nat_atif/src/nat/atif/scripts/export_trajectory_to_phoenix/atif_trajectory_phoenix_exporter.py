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
"""Phoenix exporter for complete ATIF trajectories.

See ``README.md`` in this directory for usage guidance.
"""

from __future__ import annotations

import logging
from typing import Any

from openinference.instrumentation import dangerously_using_project
from opentelemetry.sdk.resources import Resource
from phoenix.otel import HTTPSpanExporter

from nat.atif.scripts.export_trajectory_to_phoenix.atif_trajectory_exporter import ATIFTrajectorySpanExporter
from nat.plugins.opentelemetry.span_converter import convert_spans_to_otel_batch

logger = logging.getLogger(__name__)


class ATIFTrajectoryPhoenixExporter:
    """Exports ATIF trajectories to Phoenix as OpenTelemetry spans.

    Parameters
    ----------
    endpoint : str
        Phoenix server endpoint URL (e.g. ``http://localhost:6006/v1/traces``).
    project : str
        Phoenix project name for trace grouping.
    timeout : float
        HTTP request timeout in seconds.
    span_prefix : str, optional
        Prefix for NAT span attribute keys.
    """

    def __init__(
        self,
        endpoint: str,
        project: str,
        timeout: float = 60.0,
        span_prefix: str | None = None,
    ):
        self._http_exporter = HTTPSpanExporter(endpoint=endpoint, timeout=timeout)
        self._project = project
        self._resource = Resource(attributes={"openinference.project.name": project})
        self._converter = ATIFTrajectorySpanExporter(span_prefix=span_prefix)

    def export(self, trajectory_data: dict[str, Any]) -> None:
        """Convert an ATIF trajectory to spans and export to Phoenix.

        Parameters
        ----------
        trajectory_data : dict
            ATIF trajectory as a parsed JSON dict.  Must contain at
            least ``session_id``, ``agent``, and ``steps``.
        """
        nat_spans = self._converter.convert(trajectory_data)
        if not nat_spans:
            logger.warning("No spans produced from trajectory")
            return

        otel_spans = convert_spans_to_otel_batch(nat_spans)

        for span in otel_spans:
            span.set_resource(self._resource)

        try:
            with dangerously_using_project(self._project):
                self._http_exporter.export(otel_spans)  # type: ignore
            logger.info(
                "Exported %d spans for trajectory %s to Phoenix project '%s'",
                len(otel_spans),
                trajectory_data.get("session_id", "unknown"),
                self._project,
            )
        except Exception as e:
            logger.error("Error exporting trajectory spans to Phoenix: %s", e)
            raise
