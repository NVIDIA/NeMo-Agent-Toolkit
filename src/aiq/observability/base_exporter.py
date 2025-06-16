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
from abc import ABC
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.utils.reactive.subject import Subject

logger = logging.getLogger(__name__)


class AbstractExporter(ABC):
    """A base class for all telemetry exporters."""

    def __init__(self, context_state: AIQContextState | None = None):
        self._context_state = context_state or AIQContextState.get()
        self._subscription = None
        self._running = False
        self._tasks = set()  # Set of tasks created by this exporter
        self._ready_event = asyncio.Event()
        self._loop = asyncio.get_event_loop()
        self._shutdown_event = asyncio.Event()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _create_export_task(self, event: Any):
        if not self._running:
            logger.warning("%s: Attempted to create export task while not running", self.name)
            return

        # Submit the export task to the event loop and track it
        try:
            task = self._loop.create_task(self.export(event))
            # Add a name to the task for better debugging
            task.set_name(f"{self.name}_export_task")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as e:
            logger.error("%s: Failed to create export task: %s", self.name, e, exc_info=True)
            raise

    @abstractmethod
    async def export(self, trace: BaseModel) -> None:
        """Export an event."""
        pass

    @abstractmethod
    def _process_start_event(self, event: Any) -> None:
        """Process the start event."""
        pass

    @abstractmethod
    def _process_end_event(self, event: Any) -> None:
        """Process the end event."""
        pass

    @abstractmethod
    def _on_next(self, event: Any) -> None:
        """The main logic that reacts to each event."""
        pass

    def _on_error(self, exc: Exception) -> None:
        logger.error("Error in event subscription: %s", exc, exc_info=True)

    def _on_complete(self) -> None:
        logger.info("Event stream completed. No more events will arrive.")

    def _start(self) -> Subject | None:
        subject = self._context_state.event_stream.get()
        if subject is None:
            return None

        if not hasattr(subject, 'subscribe'):
            logger.error("Event stream subject does not support subscription")
            return None

        def on_next_wrapper(event: IntermediateStep) -> None:
            self._on_next(event)  # type: ignore

        self._subscription = subject.subscribe(
            on_next=on_next_wrapper,
            on_error=self._on_error,
            on_complete=self._on_complete,
        )

        self._running = True
        self._ready_event.set()
        return subject

    async def _pre_start(self):
        """Called before the listener starts."""
        pass

    @asynccontextmanager
    async def start(self):
        """Start the listener and yield control to the caller."""
        try:
            await self._pre_start()

            if self._running:
                logger.debug("Listener already running.")
                yield
                return

            subject = self._start()
            if subject is None:
                logger.warning("No event stream available.")
                yield
                return

            yield  # let the caller do their workflow

        finally:
            await self.stop()

    async def _cleanup(self):
        """Clean up any resources."""
        pass

    async def _wait_for_tasks(self, timeout: float = 5.0):
        """Wait for all tracked tasks to complete with a timeout."""
        if not self._tasks:
            return

        try:
            # Wait for all tasks to complete with a timeout
            await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("%s: Some tasks did not complete within %s seconds", self.name, timeout)
        except Exception as e:
            logger.error("%s: Error while waiting for tasks: %s", self.name, e, exc_info=True)

    async def stop(self):
        """Stop the listener."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        await self._cleanup()
        if self._subscription:
            self._subscription.unsubscribe()
        self._subscription = None

        # Create a copy of tasks to prevent modification during iteration
        tasks_to_cancel = set(self._tasks)

        self._tasks.clear()

        # Cancel only our tasks and wait for them to complete
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning("Error while canceling task %s: %s", task.get_name(), e)

        # Final check to ensure all tasks are cleaned up
        await self._wait_for_tasks()

    async def wait_ready(self):
        """Wait for the exporter to be ready."""
        await self._ready_event.wait()
