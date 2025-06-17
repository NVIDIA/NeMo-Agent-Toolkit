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

from aiq.builder.context import AIQContextState
from aiq.observability.base_exporter import AbstractExporter
from aiq.plugins.opentelemetry.otel_span import OtelSpan
from aiq.plugins.opentelemetry.span_publisher_manager import SpanPublisherManager

logger = logging.getLogger(__name__)


class AbstractOtelExporter(AbstractExporter):
    """Abstract base class for OTLP exporters.

    This class provides a base implementation for OTLP exporters.
    It handles the registration and cleanup of the OTLP span publisher.

    Args:
        context_state (AIQContextState | None): The context state to use for the exporter.
    """

    def __init__(self, context_state: AIQContextState | None = None):
        """Initialize the OTLP exporter with the specified context state."""
        super().__init__(context_state)
        self._is_shutdown = False

    def _process_start_event(self, event: OtelSpan) -> None:
        pass

    def _process_end_event(self, event: OtelSpan) -> None:
        pass

    def _on_next(self, event: OtelSpan) -> None:
        """Submit an OtelSpan to the exporter.

        Args:
            event (OtelSpan): The OtelSpan to submit.
        """

        if not isinstance(event, OtelSpan):
            logger.debug("Received unexpected event type: %s", type(event))
            return
        # Convert single span to list for consistent handling
        self._create_export_task(event)

    @abstractmethod
    async def export(self, trace: OtelSpan) -> None:
        """Export an OtelSpan.

        Args:
            trace (OtelSpan): The OtelSpan to export.
        """
        pass

    async def _pre_start(self):
        """Called before the exporter starts to ensure the exporter is registered and OTLP span publisher is started."""
        self._is_shutdown = False
        # Register exporter and start the otel span publisher
        await SpanPublisherManager.register_exporter(self._context_state)
        await SpanPublisherManager.start_publisher(self._context_state)

    async def _cleanup(self):
        """Clean up any resources.

        This method is called when the exporter is shutting down.
        It ensures that the OTLP span publisher is cleaned up and any resources are released.
        """
        if self._is_shutdown:
            return

        try:
            self._is_shutdown = True
            # Cleanup span manager and await its completion
            await SpanPublisherManager.cleanup(self._context_state)

        except Exception as e:
            logger.error("Error during exporter cleanup: %s", e)
            self._is_shutdown = True
