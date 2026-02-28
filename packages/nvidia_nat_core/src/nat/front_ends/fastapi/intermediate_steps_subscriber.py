# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import TYPE_CHECKING

from nat.builder.context import Context
from nat.data_models.api_server import ResponseATIFStep
from nat.data_models.api_server import ResponseATIFTrajectory
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.intermediate_step import IntermediateStep

if TYPE_CHECKING:
    from nat.utils.atif_converter import ATIFStreamConverter

logger = logging.getLogger(__name__)


async def pull_intermediate(_q, adapter):
    """
    Subscribes to the runner's event stream (which is now a simplified Observable)
    using direct callbacks. Processes each event with the adapter and enqueues
    results to `_q`.
    """
    intermediate_done = asyncio.Event()
    context = Context.get()
    loop = asyncio.get_running_loop()
    trace_id_emitted = False

    async def set_intermediate_done():
        intermediate_done.set()

    def on_next_cb(item: IntermediateStep):
        """
        Synchronously called whenever the runner publishes an event.
        We process it, then place it into the async queue (via a small async task).
        If adapter is None, convert the raw IntermediateStep into the complete
        ResponseIntermediateStep and place it into the queue.
        """
        nonlocal trace_id_emitted

        # Check if trace ID is now available and emit it once
        if not trace_id_emitted:
            observability_trace_id = context.observability_trace_id
            if observability_trace_id:
                from nat.data_models.api_server import ResponseObservabilityTrace
                loop.create_task(_q.put(ResponseObservabilityTrace(observability_trace_id=observability_trace_id)))
                trace_id_emitted = True

        if adapter is None:
            adapted = ResponseIntermediateStep(id=item.UUID,
                                               type=item.event_type,
                                               name=item.name or "",
                                               parent_id=item.parent_id,
                                               payload=item.payload.model_dump_json())
        else:
            adapted = adapter.process(item)

        if adapted is not None:
            loop.create_task(_q.put(adapted))

    def on_error_cb(exc: Exception):
        """
        Called if the runner signals an error. We log it and unblock our wait.
        """
        logger.error("Hit on_error: %s", exc)

        loop.create_task(set_intermediate_done())

    def on_complete_cb():
        """
        Called once the runner signals no more items. We unblock our wait.
        """
        logger.debug("Completed reading intermediate steps")

        loop.create_task(set_intermediate_done())

    # Subscribe to the runner's "reactive_event_stream" (now a simple Observable)
    _ = context.intermediate_step_manager.subscribe(on_next=on_next_cb,
                                                    on_error=on_error_cb,
                                                    on_complete=on_complete_cb)

    # Wait until on_complete or on_error sets intermediate_done
    return intermediate_done


async def pull_intermediate_atif(_q, converter: "ATIFStreamConverter"):
    """Subscribe to the IntermediateStep stream and convert to ATIF on-the-fly.

    Each time the converter flushes a complete ATIF step it is enqueued as a
    ``ResponseATIFStep``.  When the stream completes, any pending turn is
    flushed and a ``ResponseATIFTrajectory`` summary is emitted.
    """
    intermediate_done = asyncio.Event()
    context = Context.get()
    loop = asyncio.get_running_loop()
    trace_id_emitted = False

    async def set_intermediate_done():
        intermediate_done.set()

    def _enqueue_atif_step(atif_step) -> None:
        """Convert an ATIFStep into a ResponseATIFStep and enqueue it."""
        resp = ResponseATIFStep(
            step_id=atif_step.step_id,
            source=atif_step.source,
            message=atif_step.message,
            timestamp=atif_step.timestamp,
            model_name=atif_step.model_name,
            reasoning_content=atif_step.reasoning_content,
            tool_calls=[tc.model_dump() for tc in atif_step.tool_calls] if atif_step.tool_calls else None,
            observation=atif_step.observation.model_dump(exclude_none=True) if atif_step.observation else None,
            metrics=atif_step.metrics.model_dump(exclude_none=True) if atif_step.metrics else None,
            extra=atif_step.extra,
        )
        loop.create_task(_q.put(resp))

    def on_next_cb(item: IntermediateStep):
        nonlocal trace_id_emitted

        if not trace_id_emitted:
            observability_trace_id = context.observability_trace_id
            if observability_trace_id:
                from nat.data_models.api_server import ResponseObservabilityTrace
                loop.create_task(_q.put(ResponseObservabilityTrace(observability_trace_id=observability_trace_id)))
                trace_id_emitted = True

        atif_step = converter.push(item)
        if atif_step is not None:
            _enqueue_atif_step(atif_step)

    def on_error_cb(exc: Exception):
        logger.error("ATIF stream hit on_error: %s", exc)
        loop.create_task(set_intermediate_done())

    def on_complete_cb():
        logger.debug("ATIF stream complete, flushing pending turn")
        for remaining in converter.finalize():
            _enqueue_atif_step(remaining)

        trajectory = converter.get_trajectory()
        summary = ResponseATIFTrajectory(
            schema_version=trajectory.schema_version,
            session_id=trajectory.session_id,
            agent=trajectory.agent.model_dump(exclude_none=True),
            final_metrics=trajectory.final_metrics.model_dump(exclude_none=True) if trajectory.final_metrics else None,
        )
        loop.create_task(_q.put(summary))
        loop.create_task(set_intermediate_done())

    _ = context.intermediate_step_manager.subscribe(on_next=on_next_cb,
                                                    on_error=on_error_cb,
                                                    on_complete=on_complete_cb)

    return intermediate_done
