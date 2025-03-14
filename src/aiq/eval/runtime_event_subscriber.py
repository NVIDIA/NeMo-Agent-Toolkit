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

from aiq.builder.context import AIQContext
from aiq.data_models.intermediate_step import IntermediateStep

logger = logging.getLogger(__name__)


def pull_intermediate() -> asyncio.Future[list[dict]]:
    """
    Subscribes to the runner's event stream using callbacks.
    Intermediate steps are collected and, when complete, the future is set
    with the list of dumped intermediate steps.
    """
    future = asyncio.Future()
    intermediate_steps = []  # We'll store the dumped steps here.
    context = AIQContext.get()

    def on_next_cb(item: IntermediateStep):
        # Append each new intermediate step (dumped to dict) to the list.
        intermediate_steps.append(item.model_dump())

    def on_error_cb(exc: Exception):
        logger.error("Hit on_error: %s", exc)
        if not future.done():
            future.set_exception(exc)

    def on_complete_cb():
        logger.debug("Completed reading intermediate steps")
        if not future.done():
            future.set_result(intermediate_steps)

    # Subscribe with our callbacks.
    context.intermediate_step_manager.subscribe(on_next=on_next_cb, on_error=on_error_cb, on_complete=on_complete_cb)

    return future
