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

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from unittest.mock import patch

from nat.cli.commands.registry import publish as publish_mod

BUILD_SECONDS = 0.25

#: thread the fake build ran on, set by :func:`_slow_build`
build_thread: list[threading.Thread] = []


def _slow_build(package_root: str) -> MagicMock:
    """Stand-in for the real `uv build --wheel`, recording where it ran."""
    time.sleep(BUILD_SECONDS)
    build_thread.append(threading.current_thread())
    return MagicMock()


@asynccontextmanager
async def _noop_ctx(*args, **kwargs):
    yield MagicMock()


def _fake_registry() -> MagicMock:
    """Registry stub whose handler build/publish are async context managers."""
    handler = MagicMock()
    handler.build_fn = _noop_ctx
    handler.publish = _noop_ctx
    info = MagicMock()
    info.build_fn = _noop_ctx
    registry = MagicMock()
    registry.get_registry_handler.return_value = info
    return registry


async def _run_publish() -> None:
    with (
        patch("nat.cli.type_registry.GlobalTypeRegistry.get", return_value=_fake_registry()),
        patch("nat.registry_handlers.package_utils.build_artifact", _slow_build),
    ):
        await publish_mod.publish_artifact(MagicMock(), package_root=".")


async def test_publish_artifact_does_not_block_the_event_loop() -> None:
    """`nat registry publish` must not freeze the loop while the wheel builds.

    ``build_artifact()`` shells out to ``uv build --wheel``, which is slow and
    network-capable. It is synchronous, so calling it inline from
    ``publish_artifact()`` blocked the event loop for the whole build.
    """
    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.005)
            ticks += 1

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)  # baseline
    before = ticks

    await _run_publish()

    during = ticks - before
    stop.set()
    await task

    assert during > 0, (
        "event loop was starved for the whole `uv build`; build_artifact() must be offloaded with asyncio.to_thread"
    )


async def test_publish_artifact_offloads_the_build_to_a_worker_thread() -> None:
    """The build must run off the event-loop thread, not merely yield once.

    Asserting the thread is deterministic, where a heartbeat-count assertion can
    be flaky on a loaded CI worker.
    """
    build_thread.clear()

    await _run_publish()

    assert build_thread, "build_artifact() was never called; the test is not exercising the path"
    assert build_thread[0] is not threading.current_thread(), (
        "build_artifact() ran on the event loop thread; it must be offloaded with asyncio.to_thread"
    )


async def test_publish_artifact_still_reports_build_failures() -> None:
    """Offloading must not swallow the existing error handling."""

    def _boom(package_root: str):
        raise RuntimeError("wheel build failed")

    with (
        patch("nat.cli.type_registry.GlobalTypeRegistry.get", return_value=_fake_registry()),
        patch("nat.registry_handlers.package_utils.build_artifact", _boom),
    ):
        # must not raise: publish_artifact logs and returns
        await publish_mod.publish_artifact(MagicMock(), package_root=".")
