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
from aiq.observability.span.span import Span

logger = logging.getLogger(__name__)


class AbstractSpanExporter(AbstractExporter):

    def __init__(self, context_state: AIQContextState | None = None):
        super().__init__(context_state)
        self._is_shutdown = False

    def _process_start_event(self, event: Span) -> None:
        pass

    def _process_end_event(self, event: Span) -> None:
        pass

    def _on_next(self, event: Span | list[Span]) -> None:

        if not isinstance(event, Span):
            logger.debug("Received unexpected event type: %s", type(event))
            return
        # Convert single span to list for consistent handling
        self._create_export_task(event)

    @abstractmethod
    async def export(self, trace: Span) -> None:
        """Export a batch of spans."""
        pass

    async def _pre_start(self):
        """Called before the listener starts."""
        self._is_shutdown = False

    async def _cleanup(self):
        """Clean up any resources."""
        if self._is_shutdown:
            return

        try:
            self._is_shutdown = True
        except Exception as e:
            logger.error("Error during exporter cleanup: %s", e)
            self._is_shutdown = True
