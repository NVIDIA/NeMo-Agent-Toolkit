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

from nat.builder.context import ContextState
from nat.observability.exporter.base_exporter import BaseExporter

logger = logging.getLogger(__name__)


class ExporterManager:
    """
    Manages the lifecycle of asynchronous exporters.

    ExporterManager maintains a registry of exporters, allowing for dynamic addition and removal. It provides
    methods to start and stop all registered exporters concurrently, ensuring proper synchronization and
    lifecycle management. The manager is designed to prevent race conditions during exporter operations and to
    handle exporter tasks in an asyncio event loop.

    Each workflow execution gets its own ExporterManager instance to manage the lifecycle of exporters
    during that workflow's execution.

    Exporters added after `start()` is called will not be started automatically. They will only be
    started on the next lifecycle (i.e., after a stop and subsequent start).

    Args:
        shutdown_timeout (int, optional): Maximum time in seconds to wait for exporters to shut down gracefully.
        Defaults to 120 seconds.
    """

    def __init__(self, shutdown_timeout: int = 120):
        """Initialize the ExporterManager."""
        self._tasks: dict[str, asyncio.Task] = {}
        self._running: bool = False
        self._exporter_registry: dict[str, BaseExporter] = {}
        self._is_registry_shared: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._shutdown_timeout: int = shutdown_timeout
        # Track isolated exporters for proper cleanup
        self._active_isolated_exporters: dict[str, BaseExporter] = {}

    @classmethod
    def _create_with_shared_registry(cls, shutdown_timeout: int,
                                     shared_registry: dict[str, BaseExporter]) -> "ExporterManager":
        """Internal factory method for creating instances with shared registry."""
        instance = cls.__new__(cls)
        instance._tasks = {}
        instance._running = False
        instance._exporter_registry = shared_registry
        instance._is_registry_shared = True
        instance._lock = asyncio.Lock()
        instance._shutdown_event = asyncio.Event()
        instance._shutdown_timeout = shutdown_timeout
        instance._active_isolated_exporters = {}
        return instance

    def _ensure_registry_owned(self):
        """Ensure we own the registry (copy-on-write)."""
        if self._is_registry_shared:
            self._exporter_registry = self._exporter_registry.copy()
            self._is_registry_shared = False

    def add_exporter(self, name: str, exporter: BaseExporter) -> None:
        """
        Add an exporter to the manager.

        Args:
            name (str): The unique name for the exporter.
            exporter (BaseExporter): The exporter instance to add.
        """
        self._ensure_registry_owned()

        if name in self._exporter_registry:
            logger.warning("Exporter '%s' already registered. Overwriting.", name)

        self._exporter_registry[name] = exporter

    def remove_exporter(self, name: str) -> None:
        """
        Remove an exporter from the manager.

        Args:
            name (str): The name of the exporter to remove.
        """
        self._ensure_registry_owned()
        if name in self._exporter_registry:
            del self._exporter_registry[name]
        else:
            raise ValueError(f"Cannot remove exporter '{name}' because it is not registered.")

    def get_exporter(self, name: str) -> BaseExporter:
        """
        Get an exporter instance by name.

        Args:
            name (str): The name of the exporter to retrieve.

        Returns:
            BaseExporter: The exporter instance if found, otherwise raises a ValueError.

        Raises:
            ValueError: If the exporter is not found.
        """
        exporter = self._exporter_registry.get(name, None)

        if exporter is not None:
            return exporter

        raise ValueError(f"Cannot get exporter '{name}' because it is not registered.")

    async def get_all_exporters(self) -> dict[str, BaseExporter]:
        """
        Get all registered exporters instances.

        Returns:
            dict[str, BaseExporter]: A dictionary mapping exporter names to exporter instances.
        """
        return self._exporter_registry

    def create_isolated_exporters(self, context_state: ContextState | None = None) -> dict[str, BaseExporter]:
        """
        Create isolated copies of all exporters for concurrent execution.

        This uses copy-on-write to efficiently create isolated instances that share
        expensive resources but have separate mutable state.

        Args:
            context_state (ContextState | None, optional): The isolated context state for the new exporter instances.
                If not provided, a new context state will be created.

        Returns:
            dict[str, BaseExporter]: Dictionary of isolated exporter instances
        """
        # Provide default context state if None
        if context_state is None:
            context_state = ContextState.get()

        isolated_exporters = {}
        for name, exporter in self._exporter_registry.items():
            if hasattr(exporter, 'create_isolated_instance'):
                isolated_exporters[name] = exporter.create_isolated_instance(context_state)
            else:
                # Fallback for exporters that don't support isolation
                logger.warning("Exporter '%s' doesn't support isolation, using shared instance", name)
                isolated_exporters[name] = exporter
        return isolated_exporters

    async def _cleanup_isolated_exporters(self):
        """Explicitly clean up isolated exporter instances."""
        if not self._active_isolated_exporters:
            return

        logger.debug("Cleaning up %d isolated exporters", len(self._active_isolated_exporters))

        cleanup_tasks = []
        for name, exporter in self._active_isolated_exporters.items():
            try:
                # Only clean up isolated instances that have a stop method
                if hasattr(exporter, 'stop') and exporter.is_isolated_instance:
                    cleanup_tasks.append(self._cleanup_single_exporter(name, exporter))
                else:
                    logger.debug("Skipping cleanup for non-isolated exporter '%s'", name)
            except Exception as e:
                logger.exception("Error preparing cleanup for isolated exporter '%s': %s", name, e)

        if cleanup_tasks:
            # Run cleanup tasks concurrently with timeout
            try:
                await asyncio.wait_for(asyncio.gather(*cleanup_tasks, return_exceptions=True),
                                       timeout=self._shutdown_timeout)
            except asyncio.TimeoutError:
                logger.warning("Some isolated exporters did not clean up within timeout")

        self._active_isolated_exporters.clear()

    async def _cleanup_single_exporter(self, name: str, exporter: BaseExporter):
        """Clean up a single isolated exporter."""
        try:
            logger.debug("Stopping isolated exporter '%s'", name)
            await exporter.stop()
        except Exception as e:
            logger.exception("Error stopping isolated exporter '%s': %s", name, e)

    @asynccontextmanager
    async def start(self, context_state: ContextState | None = None):
        """
        Start all registered exporters concurrently.

        This method acquires a lock to ensure only one start/stop cycle is active at a time. It starts all
        currently registered exporters in their own asyncio tasks. Exporters added after this call will not be
        started until the next lifecycle.

        Args:
            context_state: Optional context state for creating isolated exporters

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

            # Create isolated exporters if context_state provided, otherwise use originals
            if context_state:
                exporters_to_start = self.create_isolated_exporters(context_state)
                # Store isolated exporters for cleanup
                self._active_isolated_exporters = exporters_to_start
                logger.debug("Created %d isolated exporters", len(exporters_to_start))
            else:
                exporters_to_start = self._exporter_registry
                # Clear isolated exporters since we're using originals
                self._active_isolated_exporters = {}

            # Start all exporters concurrently
            exporters = []
            tasks = []
            for name, exporter in exporters_to_start.items():
                task = asyncio.create_task(self._run_exporter(name, exporter))
                exporters.append(exporter)
                self._tasks[name] = task
                tasks.append(task)

            # Wait for all exporters to be ready
            await asyncio.gather(*[exporter.wait_ready() for exporter in exporters])

        try:
            yield self
        finally:
            # Clean up isolated exporters BEFORE stopping tasks
            try:
                await self._cleanup_isolated_exporters()
            except Exception as e:
                logger.exception("Error during isolated exporter cleanup: %s", e)

            # Then stop the manager tasks
            await self.stop()

    async def _run_exporter(self, name: str, exporter: BaseExporter):
        """
        Run an exporter in its own task.

        Args:
            name (str): The name of the exporter.
            exporter (BaseExporter): The exporter instance to run.
        """
        try:
            async with exporter.start():
                logger.info("Started exporter '%s'", name)
                # The context manager will keep the task alive until shutdown is signaled
                await self._shutdown_event.wait()
                logger.info("Stopped exporter '%s'", name)
        except asyncio.CancelledError:
            logger.debug("Exporter '%s' task cancelled", name)
            logger.info("Stopped exporter '%s'", name)
            raise
        except Exception as e:
            logger.error("Failed to run exporter '%s': %s", name, str(e))
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
                logger.exception("Failed to stop exporter '%s': %s", name, str(e))

        if stuck_tasks:
            logger.warning("Exporters did not shut down in time: %s", ", ".join(stuck_tasks))

    @staticmethod
    def from_exporters(exporters: dict[str, BaseExporter], shutdown_timeout: int = 120) -> "ExporterManager":
        """
        Create an ExporterManager from a dictionary of exporters.
        """
        exporter_manager = ExporterManager(shutdown_timeout=shutdown_timeout)
        for name, exporter in exporters.items():
            exporter_manager.add_exporter(name, exporter)

        return exporter_manager

    def get(self) -> "ExporterManager":
        """
        Create a copy of this ExporterManager with the same configuration using copy-on-write.

        This is the most efficient approach - shares the registry until modifications are needed.

        Returns:
            ExporterManager: A new ExporterManager instance with shared exporters (copy-on-write).
        """
        return self._create_with_shared_registry(self._shutdown_timeout, self._exporter_registry)
