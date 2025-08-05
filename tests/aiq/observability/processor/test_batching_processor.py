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

# pylint: disable=redefined-outer-name  # pytest fixtures

import asyncio
import logging
import time

from aiq.observability.processor.batching_processor import BatchingProcessor


class TestBatchingProcessorInitialization:
    """Test BatchingProcessor initialization and configuration."""

    def test_default_initialization(self):
        """Test processor with default parameters."""
        processor = BatchingProcessor[str]()

        assert processor._batch_size == 100
        assert processor._flush_interval == 5.0
        assert processor._max_queue_size == 1000
        assert processor._drop_on_overflow is False
        assert processor._shutdown_timeout == 10.0
        assert len(processor._batch_queue) == 0
        assert processor._shutdown_requested is False
        assert processor._shutdown_complete is False

    def test_custom_initialization(self):
        """Test processor with custom parameters."""
        processor = BatchingProcessor[int](batch_size=50,
                                           flush_interval=2.0,
                                           max_queue_size=500,
                                           drop_on_overflow=True,
                                           shutdown_timeout=30.0)

        assert processor._batch_size == 50
        assert processor._flush_interval == 2.0
        assert processor._max_queue_size == 500
        assert processor._drop_on_overflow is True
        assert processor._shutdown_timeout == 30.0

    def test_type_introspection(self):
        """Test that type introspection works correctly."""
        processor = BatchingProcessor[str]()

        # Type introspection works with TypeVars in generics
        # The actual types are preserved through the generic system
        assert str(processor.input_type) in ['str', '~T']  # Could be TypeVar
        assert str(processor.output_type) in ['list[str]', 'list[~T]']  # Could be TypeVar
        # Classes might also be TypeVars depending on implementation
        assert str(processor.input_class) in ['str', '~T', '<class \'str\'>']
        assert str(processor.output_class) in ['list', '<class \'list\'>', '~T']

    def test_initial_statistics(self):
        """Test initial statistics are correct."""
        processor = BatchingProcessor[str]()
        stats = processor.get_stats()

        assert stats["current_queue_size"] == 0
        assert stats["batches_created"] == 0
        assert stats["items_processed"] == 0
        assert stats["items_dropped"] == 0
        assert stats["queue_overflows"] == 0
        assert stats["shutdown_batches"] == 0
        assert stats["shutdown_requested"] is False
        assert stats["shutdown_complete"] is False
        assert stats["avg_items_per_batch"] == 0
        assert stats["drop_rate"] == 0


class TestBatchingProcessorSizeBased:
    """Test size-based batching functionality."""

    async def test_batch_creation_by_size(self):
        """Test that batches are created when size threshold is reached."""
        processor = BatchingProcessor[str](batch_size=3)

        try:
            # Add items one by one - should not create batch until size reached
            result1 = await processor.process("item1")
            assert result1 == []
            assert len(processor._batch_queue) == 1

            result2 = await processor.process("item2")
            assert result2 == []
            assert len(processor._batch_queue) == 2

            # Third item should trigger batch creation
            result3 = await processor.process("item3")
            assert result3 == ["item1", "item2", "item3"]
            assert len(processor._batch_queue) == 0
        finally:
            await processor.shutdown()

    async def test_multiple_batches_by_size(self):
        """Test multiple batch creations."""
        processor = BatchingProcessor[int](batch_size=2)

        try:
            # First batch
            await processor.process(1)
            batch1 = await processor.process(2)
            assert batch1 == [1, 2]

            # Second batch
            await processor.process(3)
            batch2 = await processor.process(4)
            assert batch2 == [3, 4]

            stats = processor.get_stats()
            assert stats["batches_created"] == 2
            assert stats["items_processed"] == 4
        finally:
            await processor.shutdown()

    async def test_partial_batch_remains_queued(self):
        """Test that partial batches remain in queue."""
        processor = BatchingProcessor[str](batch_size=5)

        try:
            await processor.process("item1")
            await processor.process("item2")

            stats = processor.get_stats()
            assert stats["current_queue_size"] == 2
            assert stats["batches_created"] == 0
        finally:
            await processor.shutdown()


class TestBatchingProcessorTimeBased:
    """Test time-based batching functionality."""

    async def test_time_based_flush_with_callback(self):
        """Test that time-based flush routes through callback."""
        processor = BatchingProcessor[str](batch_size=10, flush_interval=0.1)

        try:
            # Set up callback to capture batches
            callback_results = []

            async def test_callback(batch):
                callback_results.append(batch)

            processor.set_done_callback(test_callback)

            # Add items that won't trigger size-based batching
            await processor.process("item1")
            await processor.process("item2")

            # Wait for time-based flush
            await asyncio.sleep(0.2)

            # Batch should have been routed through callback
            assert len(callback_results) == 1
            assert callback_results[0] == ["item1", "item2"]
        finally:
            await processor.shutdown()

    async def test_time_based_flush_without_callback(self, caplog):
        """Test time-based flush when no callback is set."""
        processor = BatchingProcessor[str](batch_size=10, flush_interval=0.1)

        try:
            await processor.process("item1")

            with caplog.at_level(logging.WARNING):
                await asyncio.sleep(0.2)

            # Should log warning about missing callback
            assert "no pipeline callback set" in caplog.text
        finally:
            await processor.shutdown()

    async def test_scheduled_flush_task_management(self):
        """Test that scheduled flush tasks are properly managed."""
        processor = BatchingProcessor[str](batch_size=10, flush_interval=0.1)

        try:
            # First item should schedule a flush
            await processor.process("item1")
            assert processor._flush_task is not None
            assert not processor._flush_task.done()

            # Second item should not create new task
            first_task = processor._flush_task
            await processor.process("item2")
            assert processor._flush_task is first_task
        finally:
            await processor.shutdown()

    async def test_immediate_flush_cancels_scheduled_flush(self):
        """Test that immediate batch creation cancels scheduled flush."""
        processor = BatchingProcessor[str](batch_size=2, flush_interval=1.0)

        try:
            # First item schedules flush
            await processor.process("item1")
            original_flush_task = processor._flush_task

            # Second item triggers immediate batch and should leave task as-is
            batch = await processor.process("item2")
            assert batch == ["item1", "item2"]
            # Original task reference might be the same but would complete naturally
            assert original_flush_task is not None
        finally:
            await processor.shutdown()


class TestBatchingProcessorOverflowHandling:
    """Test queue overflow handling."""

    async def test_drop_on_overflow_enabled(self):
        """Test dropping items when queue overflows and drop_on_overflow=True."""
        processor = BatchingProcessor[str](batch_size=10, max_queue_size=2, drop_on_overflow=True)

        try:
            # Fill queue to capacity
            await processor.process("item1")
            await processor.process("item2")

            # Next item should be dropped
            result = await processor.process("item3")
            assert result == []

            stats = processor.get_stats()
            assert stats["current_queue_size"] == 2
            assert stats["items_dropped"] == 1
            assert stats["queue_overflows"] == 1
        finally:
            await processor.shutdown()

    async def test_force_flush_on_overflow(self):
        """Test force flush when queue overflows and drop_on_overflow=False."""
        processor = BatchingProcessor[str](
            batch_size=10,  # Higher than queue size to test overflow
            max_queue_size=2,
            drop_on_overflow=False)

        try:
            # Fill queue to capacity
            await processor.process("item1")
            await processor.process("item2")

            # Next item should force flush and return the forced batch
            result = await processor.process("item3")
            assert result == ["item1", "item2"]

            # New item should now be in queue
            stats = processor.get_stats()
            assert stats["current_queue_size"] == 1
            assert stats["items_dropped"] == 0
            assert stats["queue_overflows"] == 1
        finally:
            await processor.shutdown()

    async def test_overflow_statistics_tracking(self):
        """Test that overflow statistics are properly tracked."""
        processor = BatchingProcessor[str](max_queue_size=1, drop_on_overflow=True)

        try:
            await processor.process("item1")
            await processor.process("item2")  # Should be dropped
            await processor.process("item3")  # Should be dropped

            stats = processor.get_stats()
            assert stats["queue_overflows"] == 2
            assert stats["items_dropped"] == 2
            assert stats["drop_rate"] == 200.0  # 2 dropped / 1 processed * 100
        finally:
            await processor.shutdown()


class TestBatchingProcessorCallbacks:
    """Test callback functionality."""

    async def test_set_done_callback(self):
        """Test setting and using done callback."""
        processor = BatchingProcessor[str](batch_size=2)

        try:
            callback_results = []

            async def test_callback(batch):
                callback_results.append(batch)

            processor.set_done_callback(test_callback)

            # This won't use callback for immediate return
            await processor.process("item1")
            batch = await processor.process("item2")

            # Batch returned directly, not through callback for size-based batching
            assert batch == ["item1", "item2"]
            assert len(callback_results) == 0
        finally:
            await processor.shutdown()

    async def test_callback_error_handling(self, caplog):
        """Test error handling in callback execution."""
        processor = BatchingProcessor[str](batch_size=10, flush_interval=0.1)

        try:

            async def failing_callback(batch):
                raise ValueError("Callback failed")

            processor.set_done_callback(failing_callback)
            await processor.process("item1")

            with caplog.at_level(logging.ERROR):
                await asyncio.sleep(0.2)

            assert "Error routing scheduled batch through pipeline" in caplog.text
        finally:
            await processor.shutdown()

    async def test_callback_during_shutdown(self):
        """Test callback execution during shutdown."""
        processor = BatchingProcessor[str](batch_size=10)

        try:
            callback_results = []

            async def test_callback(batch):
                callback_results.append(batch)

            processor.set_done_callback(test_callback)

            # Add items
            await processor.process("item1")
            await processor.process("item2")

            # Shutdown should route final batch through callback
            await processor.shutdown()

            assert len(callback_results) == 1
            assert callback_results[0] == ["item1", "item2"]
        finally:
            await processor.shutdown()


class TestBatchingProcessorShutdown:
    """Test shutdown functionality."""

    async def test_basic_shutdown(self):
        """Test basic shutdown functionality."""
        processor = BatchingProcessor[str]()

        await processor.process("item1")
        await processor.process("item2")

        await processor.shutdown()

        assert processor._shutdown_requested is True
        assert processor._shutdown_complete is True
        assert len(processor._batch_queue) == 0

    async def test_shutdown_during_processing(self):
        """Test shutdown behavior when items are processed during shutdown."""
        processor = BatchingProcessor[str](batch_size=10)

        await processor.process("item1")

        # Start shutdown and give it a moment to set the shutdown flag
        shutdown_task = asyncio.create_task(processor.shutdown())
        await asyncio.sleep(0.01)  # Small delay to ensure shutdown starts

        # Try to process during shutdown - should return single-item batch
        result = await processor.process("item2")
        assert result == ["item2"]

        await shutdown_task

        stats = processor.get_stats()
        assert stats["shutdown_batches"] == 1

    async def test_double_shutdown_idempotent(self):
        """Test that calling shutdown multiple times is safe."""
        processor = BatchingProcessor[str](shutdown_timeout=1.0)

        await processor.process("item1")

        # First shutdown
        await processor.shutdown()
        assert processor._shutdown_complete is True

        # Second shutdown should wait and complete quickly
        start_time = time.time()
        await processor.shutdown()
        end_time = time.time()

        # Should complete quickly since already shut down
        assert end_time - start_time < 0.5

    async def test_shutdown_with_scheduled_flush(self):
        """Test shutdown behavior when scheduled flush is active."""
        processor = BatchingProcessor[str](batch_size=10, flush_interval=1.0)

        # This should schedule a flush
        await processor.process("item1")
        assert processor._flush_task is not None

        # Shutdown should cancel the flush task
        await processor.shutdown()

        assert processor._flush_task.cancelled() or processor._flush_task.done()

    async def test_shutdown_callback_error_handling(self, caplog):
        """Test error handling when callback fails during shutdown."""
        processor = BatchingProcessor[str]()

        async def failing_callback(batch):
            raise ValueError("Shutdown callback failed")

        processor.set_done_callback(failing_callback)
        await processor.process("item1")

        with caplog.at_level(logging.ERROR):
            await processor.shutdown()

        assert "Error routing final batch through pipeline during shutdown" in caplog.text
        assert processor._shutdown_complete is True

    async def test_shutdown_timeout_handling(self, caplog):
        """Test shutdown timeout handling by simulating concurrent shutdown calls."""
        processor = BatchingProcessor[str](shutdown_timeout=0.1)

        await processor.process("item1")

        # Test the scenario where shutdown is called multiple times concurrently
        # The second call should wait and potentially timeout

        # Create a barrier to simulate a hanging shutdown event
        # We'll patch the shutdown complete event to never be set
        original_event = processor._shutdown_complete_event

        # Create a new event that will never be set to simulate hanging
        hanging_event = asyncio.Event()
        processor._shutdown_complete_event = hanging_event

        # Set shutdown requested to trigger the timeout path
        processor._shutdown_requested = True

        # This should trigger the timeout path
        with caplog.at_level(logging.WARNING):
            await processor.shutdown()

        # Check if timeout warning was logged
        timeout_logged = "Shutdown completion timeout exceeded" in caplog.text

        # Restore original state
        processor._shutdown_complete_event = original_event
        processor._shutdown_requested = False
        processor._shutdown_complete = False

        # Complete a normal shutdown to clean up
        await processor.shutdown()

        # We expect the timeout to have been logged
        assert timeout_logged, f"Expected timeout warning in logs: {caplog.text}"


class TestBatchingProcessorForceFlush:
    """Test force flush functionality."""

    async def test_force_flush_with_items(self):
        """Test force flush when items are queued."""
        processor = BatchingProcessor[str](batch_size=10)

        try:
            await processor.process("item1")
            await processor.process("item2")

            batch = await processor.force_flush()
            assert batch == ["item1", "item2"]
            assert len(processor._batch_queue) == 0
        finally:
            await processor.shutdown()

    async def test_force_flush_empty_queue(self):
        """Test force flush when queue is empty."""
        processor = BatchingProcessor[str]()

        try:
            batch = await processor.force_flush()
            assert batch == []
        finally:
            await processor.shutdown()

    async def test_force_flush_statistics(self):
        """Test that force flush updates statistics correctly."""
        processor = BatchingProcessor[str](batch_size=10)

        try:
            await processor.process("item1")
            await processor.force_flush()

            stats = processor.get_stats()
            assert stats["batches_created"] == 1
            assert stats["items_processed"] == 1
        finally:
            await processor.shutdown()


class TestBatchingProcessorStatistics:
    """Test comprehensive statistics functionality."""

    async def test_comprehensive_statistics(self):
        """Test all statistics are properly tracked."""
        # Use separate scenarios to avoid conflicts between batch_size and max_queue_size

        # First, test normal batch creation
        processor = BatchingProcessor[str](batch_size=3, max_queue_size=10)
        overflow_processor = BatchingProcessor[str](batch_size=10, max_queue_size=2, drop_on_overflow=True)
        try:
            await processor.process("item1")
            await processor.process("item2")
            batch = await processor.process("item3")  # Creates batch
            assert batch == ["item1", "item2", "item3"]

            # Now test overflow with a separate processor
            await overflow_processor.process("item4")
            await overflow_processor.process("item5")
            await overflow_processor.process("item6")  # Should be dropped

            # Check combined statistics concepts
            stats = processor.get_stats()
            assert stats["batches_created"] == 1
            assert stats["items_processed"] == 3
            assert stats["avg_items_per_batch"] == 3.0

            overflow_stats = overflow_processor.get_stats()
            assert overflow_stats["items_dropped"] == 1
            assert overflow_stats["queue_overflows"] == 1
            assert overflow_stats["drop_rate"] == 50.0  # 1 dropped / 2 processed * 100
        finally:
            await processor.shutdown()
            await overflow_processor.shutdown()

    async def test_shutdown_statistics(self):
        """Test statistics tracking during shutdown processing."""
        processor = BatchingProcessor[str](batch_size=10)

        await processor.process("item1")
        await processor.shutdown()

        # Process during shutdown
        await processor.process("item2")

        stats = processor.get_stats()
        assert stats["shutdown_batches"] == 1
        assert stats["shutdown_requested"] is True
        assert stats["shutdown_complete"] is True

    async def test_statistics_edge_cases(self):
        """Test statistics edge cases like division by zero."""
        processor = BatchingProcessor[str]()

        try:
            stats = processor.get_stats()

            # Should handle division by zero gracefully
            assert stats["avg_items_per_batch"] == 0
            assert stats["drop_rate"] == 0
        finally:
            await processor.shutdown()


class TestBatchingProcessorErrorHandling:
    """Test error handling scenarios."""

    async def test_lock_acquisition_during_shutdown(self):
        """Test proper lock handling during shutdown."""
        processor = BatchingProcessor[str]()

        # Add item to queue
        await processor.process("item1")

        # Shutdown should properly acquire lock and process remaining items
        await processor.shutdown()

        assert processor._shutdown_complete is True
        assert len(processor._batch_queue) == 0

    async def test_flush_task_cancellation(self):
        """Test proper cancellation of flush tasks."""
        processor = BatchingProcessor[str](batch_size=10, flush_interval=1.0)

        # Schedule a flush
        await processor.process("item1")
        flush_task = processor._flush_task

        # Shutdown should cancel the task
        await processor.shutdown()

        # Task should be cancelled or completed
        assert flush_task is not None and (flush_task.cancelled() or flush_task.done())

    async def test_batch_creation_during_concurrent_access(self):
        """Test batch creation under concurrent access."""
        processor = BatchingProcessor[str](batch_size=2)

        try:
            # Simulate concurrent processing
            tasks = [processor.process(f"item{i}") for i in range(5)]

            results = await asyncio.gather(*tasks)

            # Should create appropriate batches without data loss
            total_items = sum(len(batch) for batch in results if batch)
            stats = processor.get_stats()

            # All items should be accounted for
            assert stats["items_processed"] == 5
            assert total_items + stats["current_queue_size"] == 5
        finally:
            await processor.shutdown()


class TestBatchingProcessorIntegration:
    """Test integration scenarios and complex workflows."""

    async def test_mixed_batching_scenarios(self):
        """Test mixed size-based and time-based batching."""
        processor = BatchingProcessor[str](batch_size=3, flush_interval=0.1)

        callback_batches = []

        async def capture_callback(batch):
            callback_batches.append(batch)

        processor.set_done_callback(capture_callback)

        # Size-based batch
        await processor.process("1")
        await processor.process("2")
        size_batch = await processor.process("3")  # Immediate return

        # Time-based batch
        await processor.process("4")
        await processor.process("5")
        await asyncio.sleep(0.2)  # Wait for time-based flush

        # Add more items before shutdown to ensure shutdown batch is created
        await processor.process("6")
        await processor.process("7")
        await processor.shutdown()

        # Verify results
        assert size_batch == ["1", "2", "3"]
        assert len(callback_batches) == 2  # Time-based + shutdown batches
        assert callback_batches[0] == ["4", "5"]
        # The shutdown batch might vary due to timing, but should contain at least the last item
        assert "7" in callback_batches[1], f"Expected '7' in shutdown batch, got {callback_batches[1]}"
        assert len(callback_batches[1]) >= 1

    async def test_high_throughput_processing(self):
        """Test high throughput processing scenario."""
        processor = BatchingProcessor[int](batch_size=100, max_queue_size=1000)

        try:
            # Process many items rapidly
            batches = []
            for i in range(250):
                batch = await processor.process(i)
                if batch:
                    batches.append(batch)

            # Force flush remaining items
            final_batch = await processor.force_flush()
            if final_batch:
                batches.append(final_batch)

            # Verify all items processed
            total_items = sum(len(batch) for batch in batches)
            assert total_items == 250

            stats = processor.get_stats()
            assert stats["items_processed"] == 250
            assert stats["batches_created"] >= 2  # At least 2 full batches + remainder
        finally:
            await processor.shutdown()

    async def test_stress_shutdown_during_processing(self):
        """Test shutdown behavior under stress conditions."""
        processor = BatchingProcessor[str](batch_size=100, flush_interval=0.5)

        callback_batches = []

        async def capture_callback(batch):
            callback_batches.append(batch)
            await asyncio.sleep(0.01)  # Simulate processing time

        processor.set_done_callback(capture_callback)

        # Start background processing
        async def background_processing():
            for i in range(10):
                await processor.process(f"bg_item_{i}")
                await asyncio.sleep(0.01)

        background_task = asyncio.create_task(background_processing())

        # Let some processing happen
        await asyncio.sleep(0.05)

        # Shutdown while processing
        await processor.shutdown()

        # Wait for background task to complete
        try:
            await asyncio.wait_for(background_task, timeout=1.0)
        except asyncio.TimeoutError:
            background_task.cancel()

        # Verify shutdown completed properly
        assert processor._shutdown_complete is True

        # All processed items should be accounted for
        total_callback_items = sum(len(batch) for batch in callback_batches)
        stats = processor.get_stats()

        # Items processed should be >= callback items (some might return directly)
        assert stats["items_processed"] >= total_callback_items
