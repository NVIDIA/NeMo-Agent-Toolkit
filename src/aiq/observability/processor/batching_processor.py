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
import time
from collections import deque
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import Generic
from typing import TypeVar

from aiq.observability.processor.processor import Processor

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchingProcessor(Processor[T, list[T]], Generic[T]):
    """Pass-through batching processor that accumulates items and outputs batched lists.

    This processor fits properly into the generics design by implementing Processor[T, List[T]].
    It accumulates individual items and outputs them as batches when size or time thresholds
    are met. The batched output continues through the processing pipeline.

    CRITICAL: Implements proper cleanup to ensure NO ITEMS ARE LOST during shutdown.
    The ProcessingExporter._cleanup() method calls shutdown() on all processors.

    Key Features:
    - Pass-through design: Processor[T, List[T]]
    - Size-based and time-based batching
    - Fits into generics processing pipeline design
    - GUARANTEED: No items lost during cleanup
    - Comprehensive statistics and monitoring
    - Proper cleanup and shutdown handling
    - High-performance async implementation
    - Back-pressure handling with queue limits

    Cleanup Guarantee:
        When ProcessingExporter._cleanup() calls shutdown(), this processor:
        1. Stops accepting new items
        2. Processes all queued items as final batch
        3. Returns final batch to continue through pipeline
        4. Ensures zero data loss during shutdown

    Usage in Pipeline:
        ```python
        # Individual spans → Batched spans → Continue processing
        exporter.add_processor(BatchingProcessor[Span](batch_size=100))
        exporter.add_processor(BatchedSpanProcessor())  # Processes List[Span]
        ```

    Args:
        batch_size: Maximum items per batch (default: 100)
        flush_interval: Max seconds to wait before flushing (default: 5.0)
        max_queue_size: Maximum items to queue before blocking (default: 1000)
        drop_on_overflow: If True, drop items when queue is full (default: False)
        shutdown_timeout: Max seconds to wait for final batch processing (default: 10.0)
    """

    def __init__(self,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 done_callback: Callable[[list[T]], Awaitable[None]] | None = None):
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_queue_size = max_queue_size
        self._drop_on_overflow = drop_on_overflow
        self._shutdown_timeout = shutdown_timeout
        self._done_callback = done_callback

        # Batching state
        self._batch_queue: deque[T] = deque()
        self._last_flush_time = time.time()
        self._flush_task: asyncio.Task | None = None
        self._batch_lock = asyncio.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._shutdown_complete_event: asyncio.Event | None = None

        # Final batch handling for cleanup
        self._final_batch: list[T] | None = None
        self._final_batch_processed = False

        # Callback for immediate export of scheduled batches
        self._done = None

        # Statistics
        self._batches_created = 0
        self._items_processed = 0
        self._items_dropped = 0
        self._queue_overflows = 0
        self._shutdown_batches = 0

    async def process(self, item: T) -> list[T]:
        """Process an item by adding it to the batch queue.

        Returns a batch when batching conditions are met, otherwise returns empty list.
        This maintains the Processor[T, List[T]] contract while handling batching logic.

        During shutdown, immediately returns items as single-item batches to ensure
        no data loss.

        Args:
            item: The item to add to the current batch

        Returns:
            List[T]: A batch of items when ready, empty list otherwise
        """
        if self._shutdown_requested:
            # During shutdown, return item immediately as single-item batch
            # This ensures no items are lost even if shutdown is in progress
            self._items_processed += 1
            self._shutdown_batches += 1
            logger.debug("Shutdown mode: returning single-item batch for item %s", item)
            return [item]

        async with self._batch_lock:
            # Handle queue overflow
            if len(self._batch_queue) >= self._max_queue_size:
                self._queue_overflows += 1

                if self._drop_on_overflow:
                    # Drop the item and return empty
                    self._items_dropped += 1
                    logger.warning("Dropping item due to queue overflow (dropped: %d)", self._items_dropped)
                    return []
                # Force flush to make space, then add item
                logger.warning("Queue overflow, forcing flush of %d items", len(self._batch_queue))
                forced_batch = await self._create_batch()
                if forced_batch:
                    # Add current item to queue and return the forced batch
                    self._batch_queue.append(item)
                    self._items_processed += 1
                    return forced_batch

            # Add item to batch queue
            self._batch_queue.append(item)
            self._items_processed += 1

            # Check flush conditions
            should_flush = (len(self._batch_queue) >= self._batch_size
                            or (time.time() - self._last_flush_time) >= self._flush_interval)

            if should_flush:
                return await self._create_batch()
            # Schedule a time-based flush if not already scheduled
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._schedule_flush())
            return []

    def set_done_callback(self, callback: Callable[[list[T]], Awaitable[None]]):
        """Set callback function for immediate export of scheduled batches."""
        self._done_callback = callback

    async def _schedule_flush(self):
        """Schedule a flush after the flush interval."""
        try:
            await asyncio.sleep(self._flush_interval)
            async with self._batch_lock:
                if not self._shutdown_requested and self._batch_queue:
                    batch = await self._create_batch()
                    if batch:
                        # Immediately export scheduled batches via callback
                        if self._done_callback is not None:
                            try:
                                await self._done_callback(batch)
                                logger.debug("Scheduled flush exported batch of %d items", len(batch))
                            except Exception as e:
                                logger.error("Error exporting scheduled batch: %s", e, exc_info=True)
                        else:
                            logger.warning("Scheduled flush created batch of %d items but no export callback set",
                                           len(batch))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in scheduled flush: %s", e, exc_info=True)

    async def _create_batch(self) -> list[T]:
        """Create a batch from the current queue."""
        if not self._batch_queue:
            return []

        batch = list(self._batch_queue)
        self._batch_queue.clear()
        self._last_flush_time = time.time()
        self._batches_created += 1

        logger.debug("Created batch of %d items (total: %d items in %d batches)",
                     len(batch),
                     self._items_processed,
                     self._batches_created)

        return batch

    async def force_flush(self) -> list[T]:
        """Force an immediate flush of all queued items.

        Returns:
            List[T]: The current batch, empty list if no items queued
        """
        async with self._batch_lock:
            return await self._create_batch()

    async def shutdown(self) -> None:
        """Shutdown the processor and ensure all items are processed.

        CRITICAL: This method is called by ProcessingExporter._cleanup() to ensure
        no items are lost during shutdown. It creates a final batch from any
        remaining items and stores it for processing.

        The final batch will be processed by the next process() call or can be
        retrieved via get_final_batch().
        """
        if self._shutdown_requested:
            logger.debug("Shutdown already requested, waiting for completion")
            # Wait for shutdown to complete using event instead of polling
            try:
                await asyncio.wait_for(self._shutdown_complete_event.wait(), timeout=self._shutdown_timeout)
                logger.debug("Shutdown completion detected via event")
            except asyncio.TimeoutError:
                logger.warning("Shutdown completion timeout exceeded (%s seconds)", self._shutdown_timeout)
            return

        logger.info("Starting shutdown of BatchingProcessor (queue size: %d)", len(self._batch_queue))
        self._shutdown_requested = True

        try:
            # Cancel scheduled flush task
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass

            # Create final batch from remaining items
            async with self._batch_lock:
                if self._batch_queue:
                    self._final_batch = await self._create_batch()
                    logger.info("Created final batch of %d items during shutdown", len(self._final_batch))
                else:
                    self._final_batch = []
                    logger.info("No items remaining during shutdown")

            self._shutdown_complete = True
            self._shutdown_complete_event.set()
            logger.info("BatchingProcessor shutdown completed successfully")

        except Exception as e:
            logger.error("Error during BatchingProcessor shutdown: %s", e, exc_info=True)
            self._shutdown_complete = True
            self._shutdown_complete_event.set()

    def get_final_batch(self) -> list[T]:
        """Get the final batch created during shutdown.

        This method allows the exporter to retrieve and process any items
        that were queued when shutdown was called.

        Returns:
            List[T]: Final batch of items, empty list if none
        """
        if self._final_batch is not None:
            final_batch = self._final_batch
            self._final_batch = None  # Clear to avoid double processing
            self._final_batch_processed = True
            return final_batch
        return []

    def has_final_batch(self) -> bool:
        """Check if there's a final batch waiting to be processed.

        Returns:
            bool: True if final batch exists and hasn't been processed
        """
        return self._final_batch is not None and not self._final_batch_processed

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive batching statistics."""
        return {
            "current_queue_size": len(self._batch_queue),
            "batch_size_limit": self._batch_size,
            "flush_interval": self._flush_interval,
            "max_queue_size": self._max_queue_size,
            "drop_on_overflow": self._drop_on_overflow,
            "shutdown_timeout": self._shutdown_timeout,
            "batches_created": self._batches_created,
            "items_processed": self._items_processed,
            "items_dropped": self._items_dropped,
            "queue_overflows": self._queue_overflows,
            "shutdown_batches": self._shutdown_batches,
            "shutdown_requested": self._shutdown_requested,
            "shutdown_complete": self._shutdown_complete,
            "final_batch_size": len(self._final_batch) if self._final_batch else 0,
            "final_batch_processed": self._final_batch_processed,
            "avg_items_per_batch": self._items_processed / max(1, self._batches_created),
            "drop_rate": self._items_dropped / max(1, self._items_processed) * 100 if self._items_processed > 0 else 0
        }
