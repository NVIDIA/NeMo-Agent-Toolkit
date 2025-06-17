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
from contextlib import asynccontextmanager

from aiq.observability.base_exporter import AbstractExporter
from aiq.observability.exporter_registry import ExporterFactory
from aiq.observability.exporter_registry import ExporterRegistry

logger = logging.getLogger(__name__)


class ExporterManager:
    """
    Manages the lifecycle of asynchronous exporters.

    ExporterManager maintains a registry of exporter factories, allowing for dynamic addition and removal. It provides
    methods to start and stop all registered exporters concurrently, ensuring proper synchronization and
    lifecycle management. The manager is designed to prevent race conditions during exporter operations and to
    handle exporter tasks in an asyncio event loop.

    Each workflow execution gets its own ExporterManager instance to manage the lifecycle of exporters
    during that workflow's execution.

    Limitations:
        - Exporter factories added after `start()` is called will not be started automatically. They will only be
        started on the next lifecycle (i.e., after a stop and subsequent start).

    Args:
        shutdown_timeout (int, optional): Maximum time in seconds to wait for exporters to shut down gracefully.
        Defaults to 120 seconds.
    """

    def __init__(self, shutdown_timeout: int = 120):
        """Initialize the ExporterManager."""
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._exporter_registry = ExporterRegistry.get_instance()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._shutdown_timeout = shutdown_timeout

    async def add_exporter(self, name: str, exporter_factory: ExporterFactory) -> None:
        """
        Add an exporter factory to the manager.

        Args:
            name (str): The unique name for the exporter.
            exporter_factory (ExporterFacotry): The exporter instance to add.
        """
        await self._exporter_registry.add(name, exporter_factory)

    async def remove_exporter(self, name: str) -> None:
        """
        Remove an exporter factory from the manager.

        Args:
            name (str): The name of the exporter to remove.
        """
        await self._exporter_registry.remove(name)

    async def get_exporter(self, name: str) -> AbstractExporter | None:
        """
        Get an exporter instance by name.

        Args:
            name (str): The name of the exporter to retrieve.

        Returns:
            AbstractExporter | None: The exporter instance if found, otherwise None.
        """
        return await self._exporter_registry.get(name)

    async def get_all_exporters(self) -> dict[str, AbstractExporter]:
        """
        Get all registered exporters instances.

        Returns:
            dict[str, AbstractExporter]: A dictionary mapping exporter names to exporter instances.
        """
        return await self._exporter_registry.get_all()

    @asynccontextmanager
    async def start(self):
        """
        Start all registered exporters concurrently.

        This method acquires a lock to ensure only one start/stop cycle is active at a time. It starts all
        currently registered exporters in their own asyncio tasks. Exporters added after this call will not be
        started until the next lifecycle.

        Yields:
            ExporterManager: The manager instance for use within the context.

        Raises:
            RuntimeError: If the manager is already running.
        """
        async with self._lock:
            if self._running:
                raise RuntimeError("Exporter manager is already running")
            self._shutdown_event.clear()
            self._running = True

            # Start all exporters concurrently
            exporters = await self.get_all_exporters()
            tasks = []
            for name, exporter in exporters.items():
                task = asyncio.create_task(self._run_exporter(name, exporter))
                self._tasks[name] = task
                tasks.append(task)

            # Wait for all exporters to be ready
            await asyncio.gather(*[exporter.wait_ready() for exporter in exporters.values()])

        try:
            yield self
        finally:
            await self.stop()

    async def _run_exporter(self, name: str, exporter: AbstractExporter):
        """
        Run an exporter in its own task.

        Args:
            name (str): The name of the exporter.
            exporter (AbstractExporter): The exporter instance to run.
        """
        try:
            async with exporter.start():
                logger.info("Started exporter '%s'", name)
                # The context manager will keep the task alive until shutdown is signaled
                await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.debug("Exporter '%s' task cancelled", name)
            raise
        except Exception as e:
            logger.error("Failed to run exporter '%s': %s", name, str(e), exc_info=True)
            # Re-raise the exception to ensure it's properly handled
            raise

    async def stop(self) -> None:
        """
        Stop all registered exporters.

        This method signals all running exporter tasks to shut down and waits for their completion, up to the
        configured shutdown timeout. If any tasks do not complete in time, a warning is logged.
        """
        async with self._lock:
            if not self._running:
                return
            self._running = False
            self._shutdown_event.set()

        # Create a copy of tasks to prevent modification during iteration
        tasks_to_cancel = dict(self._tasks)
        self._tasks.clear()
        stuck_tasks = []
        # Cancel all running tasks and await their completion
        for name, task in tasks_to_cancel.items():
            try:
                task.cancel()
                await asyncio.wait_for(task, timeout=self._shutdown_timeout)
            except asyncio.TimeoutError:
                logger.warning("Exporter '%s' task did not shut down in time and may be stuck.", name)
                stuck_tasks.append(name)
            except asyncio.CancelledError:
                logger.debug("Exporter '%s' task cancelled", name)
            except Exception as e:
                logger.error("Failed to stop exporter '%s': %s", name, str(e))

        if stuck_tasks:
            logger.warning("Exporters did not shut down in time: %s", ", ".join(stuck_tasks))
