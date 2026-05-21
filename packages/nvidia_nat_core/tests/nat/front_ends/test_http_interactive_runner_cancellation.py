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
from contextlib import asynccontextmanager

import pytest

from nat.front_ends.fastapi.execution_store import ExecutionStore
from nat.front_ends.fastapi.http_interactive_runner import HTTPInteractiveRunner


class _FakeHTTPFlowHandler:

    def set_execution_context(self, **kwargs):
        pass


class _SlowCleanupSessionManager:

    def __init__(self):
        self.cleanup_started = asyncio.Event()
        self.cleanup_finished = asyncio.Event()
        self.allow_cleanup = asyncio.Event()

    @asynccontextmanager
    async def session(self, **kwargs):
        try:
            yield object()
        finally:
            # Model slow session/workflow teardown that can happen after a streaming client disconnects.
            self.cleanup_started.set()
            await self.allow_cleanup.wait()
            self.cleanup_finished.set()


async def _workflow_that_never_finishes(_session):
    yield "first"
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_streaming_generator_close_does_not_wait_for_slow_producer_cleanup():
    session_manager = _SlowCleanupSessionManager()
    runner = HTTPInteractiveRunner(
        ExecutionStore(),
        session_manager,
        _FakeHTTPFlowHandler(),
    )
    generator = runner._streaming_generator_impl(
        None,
        workflow_gen_factory=_workflow_that_never_finishes,
        error_log_message="test streaming failure",
    )

    assert await generator.__anext__() == "data: first\n\n"

    # Simulate Starlette closing the response generator on client disconnect.
    # The close path should return promptly even while producer cleanup is still blocked.
    close_task = asyncio.create_task(generator.aclose())
    await asyncio.wait_for(session_manager.cleanup_started.wait(), timeout=1)

    close_blocked_on_session_cleanup = False
    try:
        await asyncio.wait_for(asyncio.shield(close_task), timeout=0.25)
    except TimeoutError:
        close_blocked_on_session_cleanup = True
    finally:
        session_manager.allow_cleanup.set()
        await close_task
        await asyncio.wait_for(session_manager.cleanup_finished.wait(), timeout=1)
        await asyncio.sleep(0)

    assert not close_blocked_on_session_cleanup, (
        "Closing the streaming response generator must not wait for slow producer/session cleanup. "
        "On client disconnect, Starlette closes the generator under GeneratorExit; if this path waits "
        "for session/workflow teardown, uvicorn workers can be left with partially-cancelled task state.")
