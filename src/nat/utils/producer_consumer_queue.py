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
import typing

_T = typing.TypeVar("_T")


class QueueClosed(Exception):
    'Exception raised when the queue is closed'
    pass


class AsyncIOProducerConsumerQueue(asyncio.Queue, typing.Generic[_T]):
    """
    Custom queue.Queue implementation which supports closing and uses recursive locks
    """

    def __init__(self, maxsize=0) -> None:
        super().__init__(maxsize=maxsize)

        self._closed = asyncio.Event()
        self._is_closed = False

    async def __aiter__(self):
        try:
            while True:
                yield await self.get()
        except QueueClosed:
            return

    async def join(self):
        """Block until all items in the queue have been gotten and processed.

        The count of unfinished tasks goes up whenever an item is added to the
        queue. The count goes down whenever a consumer calls task_done() to
        indicate that the item was retrieved and all work on it is complete.
        When the count of unfinished tasks drops to zero, join() unblocks.
        """

        # First wait for the closed flag to be set
        await self._closed.wait()

        if self._unfinished_tasks > 0:
            await self._finished.wait()

    async def put(self, item):
        """Put an item into the queue.

        Put an item into the queue. If the queue is full, wait until a free
        slot is available before adding item.
        """
        while self.full() and not self._is_closed:
            putter = self._get_loop().create_future()
            self._putters.append(putter)
            try:
                await putter
            except Exception:
                putter.cancel()  # Just in case putter is not done yet.
                try:
                    # Clean self._putters from canceled putters.
                    self._putters.remove(putter)
                except ValueError:
                    # The putter could be removed from self._putters by a
                    # previous get_nowait call.
                    pass
                if not self.full() and not putter.cancelled():
                    # We were woken up by get_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._putters)
                raise

        if (self._is_closed):
            raise QueueClosed  # @IgnoreException

        return self.put_nowait(item)

    async def get(self) -> _T:
        """Remove and return an item from the queue.

        If queue is empty, wait until an item is available.
        """
        while self.empty() and not self._is_closed:
            getter = self._get_loop().create_future()
            self._getters.append(getter)
            try:
                await getter
            except Exception:
                getter.cancel()  # Just in case getter is not done yet.
                try:
                    # Clean self._getters from canceled getters.
                    self._getters.remove(getter)
                except ValueError:
                    # The getter could be removed from self._getters by a
                    # previous put_nowait call.
                    pass
                if not self.empty() and not getter.cancelled():
                    # We were woken up by put_nowait(), but can't take
                    # the call.  Wake up the next in line.
                    self._wakeup_next(self._getters)
                raise

        if (self.empty() and self._is_closed):
            raise QueueClosed  # @IgnoreException

        return self.get_nowait()

    def put_blocking(self, item: _T):
        """
        Synchronously block until the item can be put.
        This method creates or uses an event loop internally to call the async put().
        If the queue is closed, it raises QueueClosed.

        NOTE: If you already have an event loop running in this same thread, calling
              `run_until_complete` can cause conflicts or an error. Typically, you only
              want to do this from a pure synchronous environment.
        """

        # If the queue is already closed, raise immediately
        if self._is_closed:
            raise QueueClosed("Queue is closed, cannot put more items.")

        # Quick check: if there's space, just put_nowait() and exit
        # (This covers the trivial case with no blocking)
        if not self.full():
            self.put_nowait(item)
            return None

        # If we do need to block, we run self.put(...) in an event loop
        # We'll attempt to get the currently running loop if there is one,
        # otherwise create a new one. If there's an existing loop, we might get
        # an error if that loop is in the same thread. Adjust logic as needed.

        try:
            # If a loop is already running in this thread, get_running_loop() will succeed.
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Means no running event loop in this thread -> create a new loop
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self.put(item))
            finally:
                loop.close()
            return None

        # If we got a running loop, but we aren't inside an async function,
        # do a "blocking" wait by scheduling the put and waiting:
        future = asyncio.run_coroutine_threadsafe(self.put(item), loop)
        result = future.result()  # blocks until done
        return result

    async def close(self):
        """Close the queue."""
        if (not self._is_closed):
            self._is_closed = True

            # Hit the flag
            self._closed.set()

            self._wakeup_next(self._putters)
            self._wakeup_next(self._getters)

    def is_closed(self) -> bool:
        """Check if the queue is closed."""
        return self._is_closed
