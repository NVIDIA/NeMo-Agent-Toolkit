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

import pytest

from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue
from nat.utils.producer_consumer_queue import QueueClosed


@pytest.mark.asyncio
@pytest.mark.parametrize("producers", [False, True])
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_close_wakes_all_waiters(producers, cancel_first):
    queue = AsyncIOProducerConsumerQueue(maxsize=1)
    if producers:
        await queue.put("buffered")
    tasks = [asyncio.create_task(queue.put(i) if producers else queue.get()) for i in range(3)]
    try:
        await asyncio.sleep(0)
        assert all(not task.done() for task in tasks)
        if cancel_first:
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
        await queue.close()
        done, pending = await asyncio.wait(tasks[1:] if cancel_first else tasks, timeout=1)
        assert not pending
        assert all(isinstance(task.exception(), QueueClosed) for task in done)
        if producers:
            assert await queue.get() == "buffered"
            queue.task_done()
        with pytest.raises(QueueClosed):
            await queue.get()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_preserves_buffered_items_and_join():
    queue = AsyncIOProducerConsumerQueue()
    await queue.put("first")
    await queue.put("second")
    await queue.close()
    await queue.close()
    assert queue.is_closed()
    with pytest.raises(QueueClosed):
        await queue.put("late")
    joined = asyncio.create_task(queue.join())
    try:
        assert [item async for item in queue] == ["first", "second"]
        await asyncio.sleep(0)
        assert not joined.done()
        queue.task_done()
        await asyncio.sleep(0)
        assert not joined.done()
        queue.task_done()
        await asyncio.wait_for(joined, timeout=1)
    finally:
        joined.cancel()
        await asyncio.gather(joined, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_empty_queue_finishes_iteration_and_join():
    queue = AsyncIOProducerConsumerQueue()
    await queue.close()
    assert [item async for item in queue] == []
    await asyncio.wait_for(queue.join(), timeout=1)
