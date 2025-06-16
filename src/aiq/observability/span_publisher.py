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
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel
from pydantic import TypeAdapter

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepState
from aiq.observability.base_exporter import AbstractExporter

logger = logging.getLogger(__name__)


class AbstractSpanPublisher(AbstractExporter):

    def __init__(self, context_state: AIQContextState | None = None):
        super().__init__(context_state)
        self._outstanding_spans: dict[str, Any] = {}
        self._span_stack: dict[str, Any] = {}
        self._stop_event = asyncio.Event()
        self._initialized = True

    def _on_next(self, event: IntermediateStep) -> None:
        """
        The main logic that reacts to each IntermediateStep.
        """
        if not isinstance(event, IntermediateStep):
            return

        if (event.event_state == IntermediateStepState.START):
            self._process_start_event(event)
        elif (event.event_state == IntermediateStepState.END):
            self._process_end_event(event)

    def _serialize_payload(self, input_value: BaseModel) -> tuple[str, bool]:
        """
        Serialize the input value to a string. Returns a tuple with the serialized value and a boolean indicating if the
        serialization is JSON or a string
        """
        try:
            return TypeAdapter(type(input_value)).dump_json(input_value).decode('utf-8'), True
        except Exception:
            # Fallback to string representation if we can't serialize using pydantic
            return str(input_value), False

    @abstractmethod
    def _process_start_event(self, event: IntermediateStep) -> None:
        pass

    @abstractmethod
    def _process_end_event(self, event: IntermediateStep):
        pass

    async def export(self, trace: Any) -> None:
        """Export spans."""
        # Push the spans into the event stream
        event_stream = self._context_state.event_stream.get()
        if event_stream:
            event_stream.on_next(trace)  # type: ignore

    async def _cleanup(self):
        """Clean up any remaining spans."""
        if self._outstanding_spans:
            logger.warning("Not all spans were closed. Remaining: %s", self._outstanding_spans)

            for span_info in self._outstanding_spans.values():
                if not span_info._ended:
                    span_info.end()

        self._outstanding_spans.clear()
        self._span_stack.clear()
        self._stop_event.set()
        self._stop_event.clear()
