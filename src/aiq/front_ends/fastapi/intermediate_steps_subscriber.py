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

import asyncio
import logging

from aiq.builder.context import Context
from aiq.data_models.api_server import ResponseIntermediateStep
from aiq.data_models.intermediate_step import IntermediateStep

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

    async def set_intermediate_done():
        intermediate_done.set()

    def on_next_cb(item: IntermediateStep):
        """
        Synchronously called whenever the runner publishes an event.
        We process it, then place it into the async queue (via a small async task).
        If adapter is None, convert the raw IntermediateStep into the complete
        AIQResponseIntermediateStep and place it into the queue.
        """
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
