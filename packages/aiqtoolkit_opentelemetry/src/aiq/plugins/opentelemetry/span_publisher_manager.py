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

from aiq.builder.context import AIQContextState
from aiq.observability.utils import AsyncSafeWeakKeyDictionary
from aiq.observability.utils import KeyedLock
from aiq.plugins.opentelemetry.otel_span_publisher import OtelSpanPublisher

logger = logging.getLogger(__name__)


class SpanPublisherManager:
    """
    Manager for the lifecycle of OtelSpanPublisher instances.

    Ensures that each context subject has at most one active OtelSpanPublisher, handles
    registration, startup, and cleanup of publishers and their associated tasks.

    Attributes:
        _instance_registry (AsyncSafeWeakKeyDictionary): A dictionary to store the state of the manager.
        _exporter_counts (AsyncSafeWeakKeyDictionary): A dictionary to store the count of exporters for each context.
        _tasks (AsyncSafeWeakKeyDictionary): A dictionary to store the tasks for each context.
        _locks (KeyedLock): A lock to synchronize access to the manager.
    """

    _instance_registry: AsyncSafeWeakKeyDictionary = AsyncSafeWeakKeyDictionary()
    _exporter_counts: AsyncSafeWeakKeyDictionary = AsyncSafeWeakKeyDictionary()
    _tasks: AsyncSafeWeakKeyDictionary = AsyncSafeWeakKeyDictionary()
    _locks: KeyedLock = KeyedLock()

    @classmethod
    async def register_exporter(cls, context_state: AIQContextState | None = None) -> None:
        """
        Register an exporter for the given context.

        Increments the exporter count for the context and ensures a publisher exists.
        If no context_state is provided, uses the current context.

        Args:
            context_state (AIQContextState, optional): The context to register the exporter for. Defaults to None.
        """
        context_state = context_state or AIQContextState.get()
        context_subject = context_state.event_stream.get()
        logger.debug("Registering exporter for context subject=%s", context_subject)

        # Update the current count of exporters for this request
        exporter_counts = await cls._exporter_counts.get(context_subject, 0)
        exporter_counts = exporter_counts or 0  # Convert None to 0
        await cls._exporter_counts.set(context_subject, exporter_counts + 1)
        _ = await cls.get_publisher(context_state)

    @classmethod
    async def get_publisher(cls, context_state: AIQContextState | None = None) -> OtelSpanPublisher:
        """
        Get or create the OtelSpanPublisher instance for the given context.

        If a publisher does not exist for the context, creates and registers one.

        Args:
            context_state (AIQContextState, optional): The context to get the publisher for. Defaults to None.

        Returns:
            OtelSpanPublisher: The publisher instance for the context.
        """
        context_state = context_state or AIQContextState.get()
        context_subject = context_state.event_stream.get()
        logger.debug("Getting publisher for context subject=%s", context_subject)

        publisher = await cls._instance_registry.get(context_subject, None)

        if publisher is not None:
            logger.debug("Found existing publisher for context subject=%s", context_subject)
            return publisher

        logger.debug("Creating new publisher for context subject=%s", context_subject)
        publisher = OtelSpanPublisher(context_state)
        await cls._instance_registry.set(context_subject, publisher)
        return publisher

    @classmethod
    async def start_publisher(cls, context_state: AIQContextState | None = None) -> OtelSpanPublisher | None:
        """
        Start the OtelSpanPublisher for the given context if there are registered exporters.

        Ensures only one publisher is running per subject. If a publisher is already running,
        it is not started again.

        Args:
            context_state (AIQContextState, optional): The context to start the publisher for. Defaults to None.

        Returns:
            OtelSpanPublisher or None: The started publisher, or None if not started.
        """
        context_state = context_state or AIQContextState.get()
        context_subject = context_state.event_stream.get()
        logger.debug("Starting publisher for context subject=%s", context_subject)

        async with cls._locks.get_lock(context_subject):
            # Check if there are any exporters registered for this subject
            exporter_counts = await cls._exporter_counts.get(context_subject, 0)
            exporter_counts = exporter_counts or 0  # Convert None to 0

            if exporter_counts > 0:
                publisher = await cls.get_publisher(context_state)

                # We only want one publisher per subject as to not produce duplicate spans
                if not publisher._running:
                    try:

                        async def _start_publisher():
                            async with publisher.start():
                                await publisher.wait_ready()
                                await publisher._stop_event.wait()

                        task = asyncio.create_task(_start_publisher())

                        await cls._tasks.set(context_subject, task)
                        logger.debug("Started publisher task for contect subject=%s", context_subject)
                        logger.info("Started exporter '%s'", publisher.name)

                    except Exception as e:
                        logger.error("Error starting publisher for context subject=%s: %s",
                                     context_subject,
                                     e,
                                     exc_info=True)
                        return None
                else:
                    logger.debug("Publisher already running for context subject=%s", context_subject)
                return publisher
            return None

    @classmethod
    async def cleanup(cls, context_state: AIQContextState | None = None) -> None:
        """
        Clean up the publisher and associated resources for a specific context.

        Cancels and awaits the publisher task, and removes all references for the context.

        Args:
            context_state (AIQContextState, optional): The context to clean up. Defaults to None.
        """
        # Gather and cancel this context's running task
        context_state = context_state or AIQContextState.get()
        context_subject = context_state.event_stream.get()
        task = await cls._tasks.get(context_subject)

        if task is not None:
            task.cancel()
            await asyncio.gather(*[task], return_exceptions=True)

        # Clear all resource registries to release references
        await cls._tasks.delete(context_subject)
        await cls._instance_registry.delete(context_subject)
        await cls._exporter_counts.delete(context_subject)
        await cls._locks.delete(context_subject)

    @classmethod
    async def cleanup_all(cls) -> None:
        """
        Clean up all publishers and associated resources for all contexts.

        Cancels and awaits all publisher tasks, and clears all registries.
        """
        # Gather and cancel all running tasks
        all_tasks: list[asyncio.Task] = await cls._tasks.values()

        for task in all_tasks:
            task.cancel()

        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Clear all resource registries to release references
        await cls._instance_registry.clear()
        await cls._exporter_counts.clear()
        await cls._tasks.clear()
        await cls._locks.clear()
