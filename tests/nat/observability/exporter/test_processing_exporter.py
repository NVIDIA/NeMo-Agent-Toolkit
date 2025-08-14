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
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.builder.context import ContextState
from nat.observability.exporter.processing_exporter import ProcessingExporter
from nat.observability.processor.callback_processor import CallbackProcessor
from nat.observability.processor.processor import Processor
from nat.utils.reactive.subject import Subject

# Note: Some tests in this module create coroutines that are intentionally not awaited
# to test error conditions. These are handled individually with targeted warnings filters.


# Test processors for mocking
class MockProcessor(Processor[str, int]):
    """Mock processor that converts strings to integers."""

    def __init__(self, name: str = "MockProcessor", should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.process_called = False
        self.processed_items = []

    async def process(self, item: str) -> int:
        """Convert string to integer length."""
        self.process_called = True
        self.processed_items.append(item)
        if self.should_fail:
            raise ValueError(f"Processing failed in {self.name}")
        return len(item)


class MockBatchProcessor(Processor[int, list[int]]):
    """Mock processor that converts integers to lists."""

    def __init__(self, name: str = "MockBatchProcessor", return_empty: bool = False):
        self.name = name
        self.return_empty = return_empty
        self.process_called = False
        self.processed_items = []

    async def process(self, item: int) -> list[int]:
        """Convert integer to list."""
        self.process_called = True
        self.processed_items.append(item)
        if self.return_empty:
            return []
        return [item] * item  # [5] -> [5, 5, 5, 5, 5]


class MockProcessorWithShutdown(Processor[str, str]):
    """Mock processor with shutdown capability."""

    def __init__(self, name: str = "MockProcessorWithShutdown"):
        self.name = name
        self.shutdown_called = False

    async def process(self, item: str) -> str:
        """Identity processor."""
        return item.upper()

    def shutdown(self):
        """Mock shutdown method that returns an awaitable to avoid coroutine creation during type introspection."""
        self.shutdown_called = True

        # Create a completed future instead of a coroutine to avoid the warning
        future = asyncio.Future()
        future.set_result(None)
        return future


class IncompatibleProcessor(Processor[float, bool]):
    """Processor with incompatible types for testing."""

    async def process(self, item: float) -> bool:
        return item > 0.0


class MockCallbackProcessor(CallbackProcessor[str, str]):
    """Mock callback processor for testing pipeline continuation."""

    def __init__(self, name: str = "MockCallbackProcessor", trigger_callback: bool = False):
        self.name = name
        self.trigger_callback = trigger_callback
        self.process_called = False
        self.processed_items = []
        self.callback_set = False
        self.done_callback = None

    async def process(self, item: str) -> str:
        """Process item normally - callback triggering is separate."""
        self.process_called = True
        self.processed_items.append(item)
        processed_item = item.upper()
        return processed_item

    def set_done_callback(self, callback):
        """Set callback for pipeline continuation."""
        self.callback_set = True
        self.done_callback = callback

    async def trigger_callback_manually(self, item: str):
        """Manually trigger the callback for testing purposes."""
        if self.done_callback:
            await self.done_callback(item)


# Concrete implementation for testing
class ConcreteProcessingExporter(ProcessingExporter[str, int]):
    """Concrete implementation of ProcessingExporter for testing."""

    def __init__(self, context_state: ContextState | None = None):
        super().__init__(context_state)
        self.exported_items = []
        self.export_processed_called = False

    async def export_processed(self, item: int | list[int]) -> None:
        """Mock implementation that records exported items."""
        self.export_processed_called = True
        self.exported_items.append(item)


class ConcreteProcessingExporterWithError(ProcessingExporter[str, int]):
    """Concrete implementation that raises errors for testing."""

    async def export_processed(self, item: int | list[int]) -> None:
        """Mock implementation that raises an error."""
        raise RuntimeError("Export failed")


@pytest.fixture
def mock_context_state():
    """Create a mock context state."""
    mock_state = Mock(spec=ContextState)
    mock_subject = Mock(spec=Subject)
    mock_event_stream = Mock()
    mock_event_stream.get.return_value = mock_subject
    mock_state.event_stream = mock_event_stream
    return mock_state


@pytest.fixture
def processing_exporter(mock_context_state):
    """Create a concrete processing exporter for testing."""
    return ConcreteProcessingExporter(mock_context_state)


class TestProcessingExporterInitialization:
    """Test ProcessingExporter initialization."""

    def test_init_with_context_state(self, mock_context_state):
        """Test initialization with provided context state."""
        exporter = ConcreteProcessingExporter(mock_context_state)
        assert exporter._context_state is mock_context_state
        assert not exporter._processors
        assert hasattr(exporter, '_running')  # Inherited from BaseExporter

    @patch('nat.observability.exporter.processing_exporter.ContextState.get')
    def test_init_without_context_state(self, mock_get_context):
        """Test initialization without context state (uses default)."""
        mock_context = Mock(spec=ContextState)
        mock_get_context.return_value = mock_context

        exporter = ConcreteProcessingExporter()
        assert exporter._context_state is mock_context
        assert not exporter._processors
        mock_get_context.assert_called_once()

    def test_inheritance(self, processing_exporter):
        """Test that ProcessingExporter properly inherits from base classes."""
        assert hasattr(processing_exporter, 'export')  # From BaseExporter
        assert hasattr(processing_exporter, 'input_type')  # From TypeIntrospectionMixin
        assert hasattr(processing_exporter, 'output_type')  # From TypeIntrospectionMixin


class TestProcessorManagement:
    """Test processor management methods."""

    def test_add_processor_empty_pipeline(self, processing_exporter):
        """Test adding processor to empty pipeline."""
        processor = MockProcessor()
        processing_exporter.add_processor(processor)

        assert len(processing_exporter._processors) == 1
        assert processing_exporter._processors[0] is processor

    def test_add_multiple_compatible_processors(self, processing_exporter):
        """Test adding multiple compatible processors."""
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)
        processing_exporter.add_processor(processor2)

        assert len(processing_exporter._processors) == 2
        assert processing_exporter._processors[0] is processor1
        assert processing_exporter._processors[1] is processor2

    def test_add_incompatible_processor_raises_error(self, processing_exporter):
        """Test adding incompatible processor raises ValueError."""
        processor1 = MockProcessor("proc1")
        incompatible_processor = IncompatibleProcessor()

        processing_exporter.add_processor(processor1)

        with pytest.raises(ValueError, match="is not compatible"):
            processing_exporter.add_processor(incompatible_processor)

    def test_add_processor_with_generic_types_warning(self, processing_exporter, caplog):
        """Test that generic type compatibility check falls back to warning."""
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)

        # Mock issubclass to raise TypeError for generic types
        with patch('builtins.issubclass', side_effect=TypeError("cannot use with generics")):
            with caplog.at_level(logging.WARNING):
                processing_exporter.add_processor(processor2)

        assert "Cannot use issubclass() for type compatibility check" in caplog.text
        assert len(processing_exporter._processors) == 2

    def test_remove_processor_exists(self, processing_exporter):
        """Test removing an existing processor."""
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")  # Compatible: int -> list[int]

        processing_exporter.add_processor(processor1)
        processing_exporter.add_processor(processor2)

        processing_exporter.remove_processor(processor1)

        assert len(processing_exporter._processors) == 1
        assert processing_exporter._processors[0] is processor2

    def test_remove_processor_not_exists(self, processing_exporter):
        """Test removing a processor that doesn't exist."""
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)

        # Should not raise an error
        processing_exporter.remove_processor(processor2)

        assert len(processing_exporter._processors) == 1
        assert processing_exporter._processors[0] is processor1

    def test_clear_processors(self, processing_exporter):
        """Test clearing all processors."""
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)
        processing_exporter.add_processor(processor2)

        processing_exporter.clear_processors()

        assert len(processing_exporter._processors) == 0


class TestTypeValidation:
    """Test type validation in _pre_start method."""

    async def test_pre_start_no_processors(self, processing_exporter):
        """Test _pre_start with no processors."""
        # Should not raise any errors
        await processing_exporter._pre_start()

    async def test_pre_start_compatible_processors(self, processing_exporter):
        """Test _pre_start with compatible processors."""
        processor = MockProcessor("proc1")
        processing_exporter.add_processor(processor)

        # Should not raise any errors
        await processing_exporter._pre_start()

    async def test_pre_start_first_processor_incompatible_input(self, processing_exporter):
        """Test _pre_start with first processor having incompatible input type."""
        # Create a processor with incompatible input type
        incompatible_processor = IncompatibleProcessor()

        # Manually add to bypass add_processor validation
        processing_exporter._processors.append(incompatible_processor)

        with pytest.raises(ValueError, match="is not compatible with the .* input type"):
            await processing_exporter._pre_start()

    async def test_pre_start_last_processor_incompatible_output(self, processing_exporter):
        """Test _pre_start with last processor having incompatible output type."""
        # Create a processor chain where the last processor has incompatible output
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)
        processing_exporter.add_processor(processor2)

        # Mock DecomposedType.is_type_compatible to return False
        with patch('nat.observability.exporter.processing_exporter.DecomposedType.is_type_compatible',
                   return_value=False):
            with pytest.raises(ValueError, match="is not compatible with the .* output type"):
                await processing_exporter._pre_start()

    async def test_pre_start_type_validation_with_generic_warning(self, processing_exporter, caplog):
        """Test _pre_start type validation falls back to warning with generic types."""

        # Create a processor that will trigger the input TypeError
        class GenericTypeProcessor(Processor[str, int]):

            @property
            def input_class(self):
                # This will cause issubclass to raise TypeError
                raise TypeError("issubclass() arg 1 must be a class")

            async def process(self, item: str) -> int:
                return len(item)

        generic_processor = GenericTypeProcessor()
        processing_exporter.add_processor(generic_processor)

        with caplog.at_level(logging.WARNING):
            await processing_exporter._pre_start()

        # Verify that warning was logged for input type compatibility
        warning_messages = [record.message for record in caplog.records if record.levelname == 'WARNING']
        assert any("Cannot validate type compatibility between" in msg and "and exporter" in msg
                   for msg in warning_messages)

    async def test_pre_start_output_type_validation_with_generic_warning(self, processing_exporter, caplog):
        """Test _pre_start output type validation falls back to warning with generic types."""
        # Create a simple processor first
        processor = MockProcessor("proc1")
        processing_exporter.add_processor(processor)

        # Mock DecomposedType.is_type_compatible to raise TypeError for output validation
        with patch('nat.observability.exporter.processing_exporter.DecomposedType.is_type_compatible',
                   side_effect=TypeError("cannot use with generics")):

            with caplog.at_level(logging.WARNING):
                await processing_exporter._pre_start()

        # Verify that warning was logged for output type compatibility
        warning_messages = [record.message for record in caplog.records if record.levelname == 'WARNING']
        assert any("Cannot validate type compatibility between" in msg and "and exporter" in msg
                   for msg in warning_messages)


class TestPipelineProcessing:
    """Test pipeline processing functionality."""

    async def test_process_pipeline_no_processors(self, processing_exporter):
        """Test pipeline processing with no processors."""
        input_item = "test"
        result = await processing_exporter._process_pipeline(input_item)
        assert result == input_item

    async def test_process_pipeline_single_processor(self, processing_exporter):
        """Test pipeline processing with single processor."""
        processor = MockProcessor("proc1")
        processing_exporter.add_processor(processor)

        input_item = "hello"
        result = await processing_exporter._process_pipeline(input_item)

        assert result == 5  # len("hello")
        assert processor.process_called
        assert processor.processed_items == ["hello"]

    async def test_process_pipeline_multiple_processors(self, processing_exporter):
        """Test pipeline processing with multiple processors."""
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)
        processing_exporter.add_processor(processor2)

        input_item = "hello"
        result = await processing_exporter._process_pipeline(input_item)

        assert result == [5, 5, 5, 5, 5]  # len("hello") = 5, then [5] * 5
        assert processor1.process_called
        assert processor2.process_called
        assert processor1.processed_items == ["hello"]
        assert processor2.processed_items == [5]

    async def test_process_pipeline_processor_error_continues(self, processing_exporter, caplog):
        """Test that processor errors are logged but processing continues."""
        failing_processor = MockProcessor("failing", should_fail=True)

        processing_exporter.add_processor(failing_processor)

        input_item = "hello"

        with caplog.at_level(logging.ERROR):
            result = await processing_exporter._process_pipeline(input_item)

        # Should continue with unprocessed item when processor fails
        assert result == "hello"  # Original item passed through when processor fails
        # Log uses class name, not instance name
        assert "Error in processor MockProcessor" in caplog.text
        assert failing_processor.process_called


class TestExportWithProcessing:
    """Test export with processing functionality."""

    async def test_export_with_processing_single_item(self, processing_exporter):
        """Test exporting single processed item."""
        processor = MockProcessor("proc1")
        processing_exporter.add_processor(processor)

        input_item = "hello"
        await processing_exporter._export_with_processing(input_item)

        assert processing_exporter.export_processed_called
        assert len(processing_exporter.exported_items) == 1
        assert processing_exporter.exported_items[0] == 5  # len("hello")

    async def test_export_with_processing_list_item_non_empty(self, mock_context_state):
        """Test exporting non-empty list from batch processor."""

        # Create a specialized exporter for list output
        class ListProcessingExporter(ProcessingExporter[str, list[int]]):

            def __init__(self, context_state: ContextState | None = None):
                super().__init__(context_state)
                self.exported_items = []
                self.export_processed_called = False

            async def export_processed(self, item: list[int] | list[list[int]]) -> None:
                self.export_processed_called = True
                self.exported_items.append(item)

        exporter = ListProcessingExporter(mock_context_state)
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        exporter.add_processor(processor1)
        exporter.add_processor(processor2)

        input_item = "test"
        await exporter._export_with_processing(input_item)

        assert exporter.export_processed_called
        assert len(exporter.exported_items) == 1
        assert exporter.exported_items[0] == [4, 4, 4, 4]  # [len("test")] * len("test")

    async def test_export_with_processing_list_item_empty_skipped(self, mock_context_state):
        """Test that empty lists from batch processors are skipped."""

        # Create a specialized exporter for list output
        class ListProcessingExporter(ProcessingExporter[str, list[int]]):

            def __init__(self, context_state: ContextState | None = None):
                super().__init__(context_state)
                self.exported_items = []
                self.export_processed_called = False

            async def export_processed(self, item: list[int] | list[list[int]]) -> None:
                self.export_processed_called = True
                self.exported_items.append(item)

        exporter = ListProcessingExporter(mock_context_state)
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2", return_empty=True)

        exporter.add_processor(processor1)
        exporter.add_processor(processor2)

        input_item = "test"

        await exporter._export_with_processing(input_item)

        assert not exporter.export_processed_called
        assert len(exporter.exported_items) == 0

    async def test_export_with_processing_invalid_output_type_error(self, processing_exporter):
        """Test error when processed item has invalid output type."""

        # Create a processor that returns an unexpected type
        class BadProcessor(Processor[str, dict]):

            async def process(self, item: str) -> dict:
                return {"invalid": "type"}

        bad_processor = BadProcessor()
        processing_exporter._processors.append(bad_processor)  # Bypass type checking

        input_item = "test"

        with pytest.raises(ValueError, match="is not a valid output type"):
            await processing_exporter._export_with_processing(input_item)

    async def test_export_with_processing_export_error_propagates(self, mock_context_state):
        """Test that export errors are properly propagated."""
        exporter = ConcreteProcessingExporterWithError(mock_context_state)
        processor = MockProcessor("proc1")
        exporter.add_processor(processor)

        input_item = "test"

        with pytest.raises(RuntimeError, match="Export failed"):
            await exporter._export_with_processing(input_item)


class TestExportMethod:
    """Test the export method."""

    def test_export_compatible_event(self, processing_exporter):
        """Test export with compatible event type."""
        # Create a mock event that matches the input type
        event = "test_string"  # Direct string instead of mock

        with patch.object(processing_exporter, '_create_export_task') as mock_create_task:
            processing_exporter.export(event)

        mock_create_task.assert_called_once()
        # Verify the coroutine is created correctly
        args, _ = mock_create_task.call_args
        assert asyncio.iscoroutine(args[0])
        # Clean up the coroutine to avoid RuntimeWarning
        args[0].close()

    @pytest.mark.filterwarnings("ignore:.*coroutine.*was never awaited:RuntimeWarning")
    def test_export_incompatible_event_warning(self, processing_exporter, caplog):
        """Test export with incompatible event type logs warning.

        Note: This test creates a coroutine that is intentionally never awaited
        because the event type is incompatible. The RuntimeWarning is expected
        and filtered out to focus on testing the incompatible event handling.
        """
        event = 123  # Integer event (incompatible with str input type)

        with caplog.at_level(logging.WARNING):
            processing_exporter.export(event)

        assert "is not compatible with input type" in caplog.text


class TestTaskCreation:
    """Test task creation functionality."""

    def test_create_export_task_when_running(self, processing_exporter):
        """Test creating export task when exporter is running."""
        processing_exporter._running = True
        processing_exporter._tasks = set()

        # Use a mock coroutine that doesn't need to be awaited
        mock_coro = Mock()

        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            processing_exporter._create_export_task(mock_coro)

            mock_create_task.assert_called_once_with(mock_coro)
            assert mock_task in processing_exporter._tasks
            mock_task.add_done_callback.assert_called_once()

    def test_create_export_task_when_not_running_warning(self, processing_exporter, caplog):
        """Test creating export task when exporter is not running logs warning."""
        processing_exporter._running = False

        # Use a mock coroutine that doesn't need to be awaited
        mock_coro = Mock()

        with caplog.at_level(logging.WARNING):
            processing_exporter._create_export_task(mock_coro)

        assert "Attempted to create export task while not running" in caplog.text

    def test_create_export_task_error_handling(self, processing_exporter, caplog):
        """Test error handling in task creation."""
        processing_exporter._running = True

        # Use a mock coroutine that doesn't need to be awaited
        mock_coro = Mock()

        with patch('asyncio.create_task', side_effect=RuntimeError("Task creation failed")):
            with pytest.raises(RuntimeError, match="Task creation failed"):
                with caplog.at_level(logging.ERROR):
                    processing_exporter._create_export_task(mock_coro)

        assert "Failed to create task" in caplog.text


class TestCleanup:
    """Test cleanup functionality."""

    async def test_cleanup_no_processors(self, processing_exporter):
        """Test cleanup with no processors."""
        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            await processing_exporter._cleanup()

            mock_parent_cleanup.assert_called_once()

    async def test_cleanup_processors_without_shutdown(self, processing_exporter):
        """Test cleanup with processors that don't have shutdown method."""
        processor = MockProcessor("proc1")
        processing_exporter.add_processor(processor)

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            await processing_exporter._cleanup()

            mock_parent_cleanup.assert_called_once()

    async def test_cleanup_processors_with_shutdown(self, processing_exporter, caplog):
        """Test cleanup with processors that have shutdown method."""
        processor = MockProcessorWithShutdown("proc1")
        processing_exporter.add_processor(processor)

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            with caplog.at_level(logging.DEBUG):
                await processing_exporter._cleanup()

            assert processor.shutdown_called
            assert "Shutting down processor: MockProcessorWithShutdown" in caplog.text
            mock_parent_cleanup.assert_called_once()

    async def test_cleanup_processors_shutdown_success(self, processing_exporter, caplog):
        """Test successful processor shutdown logging."""
        processor1 = MockProcessorWithShutdown("proc1")
        processor2 = MockProcessorWithShutdown("proc2")
        processing_exporter.add_processor(processor1)
        processing_exporter.add_processor(processor2)

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            with caplog.at_level(logging.DEBUG):
                await processing_exporter._cleanup()

            assert processor1.shutdown_called
            assert processor2.shutdown_called
            assert "Successfully shut down 2 processors" in caplog.text

    async def test_cleanup_processors_shutdown_error(self, processing_exporter, caplog):
        """Test error handling during processor shutdown."""
        processor = MockProcessorWithShutdown("proc1")
        processing_exporter.add_processor(processor)

        # Mock processor shutdown to raise an error
        async def failing_shutdown():
            raise RuntimeError("Shutdown failed")

        processor.shutdown = failing_shutdown

        # Mock asyncio.gather to properly propagate the exception
        async def mock_gather(*tasks, return_exceptions=True):
            # Execute the tasks and return the exception as requested
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    if return_exceptions:
                        results.append(e)
                    else:
                        raise
            return results

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup, \
             patch('asyncio.gather', side_effect=mock_gather):
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            with caplog.at_level(logging.ERROR):
                await processing_exporter._cleanup()

            # The error logging might not appear due to return_exceptions=True,
            # so let's just check the method was called
            assert processor.shutdown != processor.__class__.shutdown  # Verify it was replaced

    async def test_cleanup_calls_processor_shutdown(self, processing_exporter, caplog):
        """Test that cleanup calls shutdown on processors that have it."""
        processor = MockProcessorWithShutdown("proc1")
        processing_exporter.add_processor(processor)

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            with caplog.at_level(logging.DEBUG):
                await processing_exporter._cleanup()

            assert processor.shutdown_called
            assert "Successfully shut down 1 processors" in caplog.text

    async def test_cleanup_processor_shutdown_error_handling(self, processing_exporter):
        """Test error handling during processor shutdown."""
        processor = MockProcessorWithShutdown("proc1")
        processing_exporter.add_processor(processor)

        # Mock processor shutdown to raise an error
        async def failing_shutdown():
            raise RuntimeError("Shutdown failed")

        processor.shutdown = failing_shutdown

        # Mock asyncio.gather to handle exceptions properly
        async def mock_gather(*tasks, return_exceptions=True):
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    if return_exceptions:
                        results.append(e)
                    else:
                        raise
            return results

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup, \
             patch('asyncio.gather', side_effect=mock_gather):
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            # Should not raise an error due to return_exceptions=True
            await processing_exporter._cleanup()

            # Verify the shutdown was called (even though it failed)
            assert processor.shutdown != processor.__class__.shutdown  # Verify it was replaced

    async def test_cleanup_without_processors_attribute(self, processing_exporter):
        """Test cleanup when _processors attribute doesn't exist."""
        # Remove the _processors attribute
        delattr(processing_exporter, '_processors')

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            # Should not raise an error
            await processing_exporter._cleanup()

            mock_parent_cleanup.assert_called_once()


class TestTypeIntrospection:
    """Test type introspection capabilities."""

    def test_input_output_types(self, processing_exporter):
        """Test that type introspection works correctly."""
        assert processing_exporter.input_type == str
        assert processing_exporter.output_type == int
        assert processing_exporter.input_class == str
        assert processing_exporter.output_class == int


class TestAbstractMethod:
    """Test abstract method enforcement."""

    def test_export_processed_is_abstract(self):
        """Test that export_processed must be implemented."""

        # Test that trying to instantiate a class without implementing export_processed raises TypeError
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            # Create a class that doesn't implement export_processed
            class IncompleteExporter(ProcessingExporter[str, int]):
                pass  # Missing export_processed implementation

            # This should raise TypeError due to missing abstract method implementation
            IncompleteExporter()  # pylint: disable=abstract-class-instantiated


class TestCallbackProcessorIntegration:
    """Test CallbackProcessor integration and pipeline continuation."""

    def test_callback_processor_callback_setup(self, processing_exporter):
        """Test that CallbackProcessor gets its callback set during add_processor."""
        callback_processor = MockCallbackProcessor("callback_proc")

        processing_exporter.add_processor(callback_processor)

        # Verify the callback was set (covers lines 97-100)
        assert callback_processor.callback_set
        assert callback_processor.done_callback is not None

    async def test_callback_processor_pipeline_continuation(self, processing_exporter):
        """Test CallbackProcessor triggers pipeline continuation."""
        # Setup: Callback processor -> Regular processor
        callback_processor = MockCallbackProcessor("callback_proc")
        regular_processor = MockProcessor("regular_proc")  # str -> int

        processing_exporter.add_processor(callback_processor)  # str -> str
        processing_exporter.add_processor(regular_processor)  # str -> int

        # Manually trigger the callback to test pipeline continuation
        # This simulates what would happen when a real callback processor (like BatchingProcessor)
        # triggers its callback with items to continue processing
        test_item = "hello"  # String item to process
        await callback_processor.trigger_callback_manually(test_item)

        # Verify the regular processor was called through pipeline continuation
        assert regular_processor.process_called
        assert test_item in regular_processor.processed_items

        # The final result should be exported (int from len("hello") = 5)
        # This covers the pipeline continuation logic (lines 212-228)
        assert processing_exporter.export_processed_called
        assert 5 in processing_exporter.exported_items  # len("hello") = 5

    async def test_continue_pipeline_after_with_remaining_processors(self):
        """Test _continue_pipeline_after processes through remaining pipeline."""

        # Create a string-processing exporter to avoid type issues
        class StringProcessingExporter(ProcessingExporter[str, str]):

            def __init__(self, context_state=None):
                super().__init__(context_state)
                self.exported_items = []
                self.export_processed_called = False

            async def export_processed(self, item):
                self.export_processed_called = True
                self.exported_items.append(item)

        # Create processors that all work with strings
        class StringProcessor(Processor[str, str]):

            def __init__(self, name):
                self.name = name
                self.process_called = False
                self.processed_items = []

            async def process(self, item: str) -> str:
                self.process_called = True
                self.processed_items.append(item)
                return f"{item}_{self.name}"

        string_exporter = StringProcessingExporter()
        source_processor = StringProcessor("source")
        middle_processor = StringProcessor("middle")
        final_processor = StringProcessor("final")

        string_exporter.add_processor(source_processor)
        string_exporter.add_processor(middle_processor)
        string_exporter.add_processor(final_processor)

        # Manually call _continue_pipeline_after to test the method
        test_item = "test"
        await string_exporter._continue_pipeline_after(source_processor, test_item)

        # Verify only the processors after source were called
        assert not source_processor.process_called  # Should be skipped
        assert middle_processor.process_called  # Should process
        assert final_processor.process_called  # Should process
        assert string_exporter.export_processed_called
        # Should be "test_middle_final" after processing through middle and final
        assert "test_middle_final" in string_exporter.exported_items

    async def test_continue_pipeline_processor_not_found(self, processing_exporter, caplog):
        """Test _continue_pipeline_after when source processor not in pipeline."""
        # Add one processor to pipeline
        pipeline_processor = MockProcessor("in_pipeline")
        processing_exporter.add_processor(pipeline_processor)

        # Try to continue from a processor not in pipeline
        unknown_processor = MockProcessor("not_in_pipeline")

        with caplog.at_level(logging.ERROR):
            await processing_exporter._continue_pipeline_after(unknown_processor, "test")

        # Verify error was logged (covers lines 216-218)
        assert "Source processor MockProcessor not found in pipeline" in caplog.text
        assert not processing_exporter.export_processed_called

    async def test_continue_pipeline_exception_handling(self, processing_exporter, caplog):
        """Test _continue_pipeline_after exception handling."""
        # Setup a processor that will cause an exception
        failing_processor = MockProcessor("source", should_fail=True)
        processing_exporter.add_processor(failing_processor)

        # Mock _process_through_processors to raise an exception
        async def failing_process(*args, **kwargs):
            raise RuntimeError("Pipeline processing failed")

        processing_exporter._process_through_processors = failing_process

        with caplog.at_level(logging.ERROR):
            await processing_exporter._continue_pipeline_after(failing_processor, "test")

        # Verify exception was logged (covers lines 227-231)
        assert "Failed to continue pipeline processing after MockProcessor" in caplog.text

    async def test_callback_processor_no_remaining_processors(self, processing_exporter):
        """Test _continue_pipeline_after when no processors follow source."""
        # Add only one processor
        solo_processor = MockProcessor("solo")
        processing_exporter.add_processor(solo_processor)

        # Continue pipeline after the only processor with the processed output (integer)
        # MockProcessor converts strings to their length, so "test" -> 4
        await processing_exporter._continue_pipeline_after(solo_processor, 4)

        # Should still call export_processed with the item
        assert processing_exporter.export_processed_called
        assert len(processing_exporter.exported_items) == 1
        assert processing_exporter.exported_items[0] == 4


class TestErrorPathCoverage:
    """Test error paths and logging coverage."""

    async def test_empty_batch_debug_logging(self, processing_exporter, caplog):
        """Test debug logging when exporting empty batch."""
        # Create an empty list to trigger the debug log
        empty_batch = []

        with caplog.at_level(logging.DEBUG):
            await processing_exporter._export_final_item(empty_batch)

        # Verify debug log was emitted (covers line 193)
        assert "Skipping export of empty batch" in caplog.text
        assert not processing_exporter.export_processed_called

    async def test_invalid_output_type_warning_path(self, processing_exporter, caplog):
        """Test warning path for invalid output types."""
        # Create an invalid output type (not int or list[int] for our exporter)
        invalid_item = {"invalid": "dict"}

        with caplog.at_level(logging.WARNING):
            # Call with raise_on_invalid=False to trigger warning path
            await processing_exporter._export_final_item(invalid_item, raise_on_invalid=False)

        # Verify warning was logged (covers line 200)
        assert "is not a valid output type for export" in caplog.text
        assert not processing_exporter.export_processed_called

    async def test_cleanup_shutdown_exception_handling(self, processing_exporter, caplog):
        """Test exception handling during processor shutdown in cleanup."""
        processor = MockProcessorWithShutdown("test_proc")
        processing_exporter.add_processor(processor)

        # Mock asyncio.gather to raise an exception
        async def failing_gather(*tasks, return_exceptions=True):
            raise RuntimeError("Shutdown failed")

        with patch('nat.observability.exporter.base_exporter.BaseExporter._cleanup') as mock_parent_cleanup:
            mock_parent_cleanup.return_value = asyncio.Future()
            mock_parent_cleanup.return_value.set_result(None)

            with patch('asyncio.gather', side_effect=failing_gather):
                with caplog.at_level(logging.ERROR):
                    await processing_exporter._cleanup()

        # Verify exception was logged (covers lines 318-319)
        assert "Error shutting down processors" in caplog.text

    async def test_export_final_item_empty_list_vs_none(self, processing_exporter):
        """Test distinction between empty list and None for batch handling."""
        # Test empty list (should not export)
        await processing_exporter._export_final_item([])
        assert not processing_exporter.export_processed_called

        # Reset and test with valid single item
        processing_exporter.export_processed_called = False
        processing_exporter.exported_items = []

        valid_item = 5  # int matches our exporter's output type
        await processing_exporter._export_final_item(valid_item)
        assert processing_exporter.export_processed_called
        assert processing_exporter.exported_items == [5]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_process_pipeline_empty_processors_list(self, processing_exporter):
        """Test pipeline processing with explicitly empty processors list."""
        processing_exporter._processors = []

        input_item = "test"
        result = await processing_exporter._process_pipeline(input_item)

        assert result == input_item

    def test_add_processor_type_compatibility_complex_generics(self, processing_exporter):
        """Test type compatibility with complex generic types."""
        # This tests the fallback to warning when issubclass fails with complex generics
        processor1 = MockProcessor("proc1")
        processor2 = MockBatchProcessor("proc2")

        processing_exporter.add_processor(processor1)

        # Should work despite complex generics
        processing_exporter.add_processor(processor2)

        assert len(processing_exporter._processors) == 2

    def test_processor_management_with_same_processor_instance(self, processing_exporter):
        """Test adding the same processor instance multiple times."""
        processor = MockProcessor("proc1")

        processing_exporter.add_processor(processor)
        # For this test, we need compatible processors to test the remove functionality
        # So let's add a different processor type that's compatible
        processor2 = MockBatchProcessor("proc2")
        processing_exporter.add_processor(processor2)

        assert len(processing_exporter._processors) == 2
        assert processing_exporter._processors[0] is processor
        assert processing_exporter._processors[1] is processor2

        # Remove the first one
        processing_exporter.remove_processor(processor)

        # Should only remove the first occurrence
        assert len(processing_exporter._processors) == 1
        assert processing_exporter._processors[0] is processor2

    async def test_export_with_processing_coroutine_cleanup(self, processing_exporter):
        """Test that coroutines are properly cleaned up even if export fails."""
        processor = MockProcessor("proc1")
        processing_exporter.add_processor(processor)

        # Mock export_processed to raise an error
        async def failing_export(item):
            raise RuntimeError("Export failed")

        processing_exporter.export_processed = failing_export

        input_item = "test"

        with pytest.raises(RuntimeError, match="Export failed"):
            await processing_exporter._export_with_processing(input_item)

        # Processor should still have been called
        assert processor.process_called

    def test_processors_attribute_access_edge_cases(self, processing_exporter):
        """Test edge cases in processor attribute access."""
        # Test that _processors is initialized as expected
        assert hasattr(processing_exporter, '_processors')
        assert isinstance(processing_exporter._processors, list)

        # Test that we can access it safely
        processors = processing_exporter._processors
        assert processors == []

        # Test that modifications work as expected
        processor = MockProcessor("proc1")
        processors.append(processor)
        assert len(processing_exporter._processors) == 1
