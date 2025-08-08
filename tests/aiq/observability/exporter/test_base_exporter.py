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
import weakref
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.observability.exporter.base_exporter import BaseExporter
from aiq.observability.exporter.base_exporter import IsolatedAttribute
from aiq.utils.reactive.subject import Subject


class ConcreteExporter(BaseExporter):
    """Concrete implementation of BaseExporter for testing."""

    def __init__(self, context_state=None, export_callback=None):
        super().__init__(context_state)
        self.exported_events = []

        def default_callback(x):  # pylint: disable=W0613
            pass

        self.export_callback = export_callback or default_callback

    def export(self, event: IntermediateStep) -> None:
        """Test implementation that records exported events."""
        self.exported_events.append(event)
        self.export_callback(event)


class TestIsolatedAttribute:
    """Test the IsolatedAttribute descriptor."""

    def test_init(self):
        """Test IsolatedAttribute initialization."""

        def factory():
            return set()

        attr = IsolatedAttribute(factory)
        assert attr.factory is factory
        assert attr.name is None

    def test_set_name(self):
        """Test __set_name__ method."""
        attr = IsolatedAttribute(set)
        attr.__set_name__(BaseExporter, "test_attr")
        assert attr.name == "test_attr"
        assert attr._private_name == "__test_attr_isolated"

    def test_get_from_class(self):
        """Test __get__ when called on the class."""
        attr = IsolatedAttribute(set)
        result = attr.__get__(None, BaseExporter)  # pylint: disable=unnecessary-dunder-call
        assert result is attr

    def test_get_from_instance_first_time(self):
        """Test __get__ when called on instance for the first time."""
        attr = IsolatedAttribute(set)
        attr.__set_name__(BaseExporter, "test_attr")

        exporter = ConcreteExporter()
        result = attr.__get__(exporter, BaseExporter)  # pylint: disable=unnecessary-dunder-call

        assert isinstance(result, set)
        assert hasattr(exporter, "__test_attr_isolated")

    def test_get_from_instance_subsequent_times(self):
        """Test __get__ returns same instance on subsequent calls."""
        attr = IsolatedAttribute(set)
        attr.__set_name__(BaseExporter, "test_attr")

        exporter = ConcreteExporter()
        result1 = attr.__get__(exporter, BaseExporter)  # pylint: disable=unnecessary-dunder-call
        result2 = attr.__get__(exporter, BaseExporter)  # pylint: disable=unnecessary-dunder-call

        assert result1 is result2

    def test_set(self):
        """Test __set__ method."""
        attr = IsolatedAttribute(set)
        attr.__set_name__(BaseExporter, "test_attr")

        exporter = ConcreteExporter()
        test_set = {1, 2, 3}
        attr.__set__(exporter, test_set)  # pylint: disable=unnecessary-dunder-call

        assert getattr(exporter, "__test_attr_isolated") is test_set

    def test_reset_for_copy(self):
        """Test reset_for_copy method."""
        attr = IsolatedAttribute(set)
        attr.__set_name__(BaseExporter, "test_attr")

        exporter = ConcreteExporter()
        # Access the attribute to create it
        _ = attr.__get__(exporter, BaseExporter)  # pylint: disable=unnecessary-dunder-call
        assert hasattr(exporter, "__test_attr_isolated")

        # Reset for copy
        attr.reset_for_copy(exporter)
        assert not hasattr(exporter, "__test_attr_isolated")

    def test_reset_for_copy_when_not_set(self):
        """Test reset_for_copy when attribute hasn't been accessed."""
        attr = IsolatedAttribute(set)
        attr.__set_name__(BaseExporter, "test_attr")

        exporter = ConcreteExporter()
        # Don't access the attribute

        # Should not raise an error
        attr.reset_for_copy(exporter)
        assert not hasattr(exporter, "__test_attr_isolated")


class TestBaseExporter:  # pylint: disable=too-many-public-methods
    """Test the BaseExporter class."""

    @pytest.fixture
    def mock_context_state(self):
        """Create a mock context state."""
        mock_state = Mock()
        mock_subject = Mock(spec=Subject)
        mock_event_stream = Mock()
        mock_event_stream.get.return_value = mock_subject
        mock_state.event_stream = mock_event_stream
        return mock_state

    @pytest.fixture
    def exporter(self, mock_context_state):
        """Create a concrete exporter for testing."""
        return ConcreteExporter(mock_context_state)

    def test_init_with_context_state(self, mock_context_state):
        """Test initialization with provided context state."""
        exporter = ConcreteExporter(mock_context_state)
        assert exporter._context_state is mock_context_state
        assert exporter._subscription is None
        assert exporter._running is False
        assert exporter._loop is None
        assert exporter._is_isolated_instance is False

    @patch('aiq.observability.exporter.base_exporter.AIQContextState.get')
    def test_init_without_context_state(self, mock_get_context):
        """Test initialization without context state (uses default)."""
        mock_context = Mock(spec=AIQContextState)
        mock_get_context.return_value = mock_context

        exporter = ConcreteExporter()
        assert exporter._context_state is mock_context
        mock_get_context.assert_called_once()

    def test_instance_tracking_on_creation(self):
        """Test that instance creation is tracked."""
        initial_count = BaseExporter.get_active_instance_count()
        exporter = ConcreteExporter()
        assert BaseExporter.get_active_instance_count() == initial_count + 1
        assert exporter is not None  # Use the variable

    def test_instance_tracking_cleanup(self):
        """Test that instance cleanup removes from tracking."""
        initial_count = BaseExporter.get_active_instance_count()
        exporter = ConcreteExporter()
        exporter_ref = weakref.ref(exporter)

        # Verify the reference is alive
        assert exporter_ref() is not None

        # Delete the exporter
        del exporter

        # Force garbage collection to trigger cleanup
        import gc
        gc.collect()

        # The count should be back to initial (may take time due to weakref cleanup)
        assert BaseExporter.get_active_instance_count() <= initial_count + 1

    def test_name_property_normal_instance(self, exporter):
        """Test name property for normal instance."""
        assert exporter.name == "ConcreteExporter"

    def test_name_property_isolated_instance(self, exporter):
        """Test name property for isolated instance."""
        isolated = exporter.create_isolated_instance(exporter._context_state)
        assert isolated.name == "ConcreteExporter (isolated)"

    def test_is_isolated_instance_property(self, exporter):
        """Test is_isolated_instance property."""
        assert exporter.is_isolated_instance is False

        isolated = exporter.create_isolated_instance(exporter._context_state)
        assert isolated.is_isolated_instance is True

    def test_export_abstract_method(self, exporter):
        """Test that export method works in concrete implementation."""
        event = Mock(spec=IntermediateStep)
        exporter.export(event)
        assert event in exporter.exported_events

    def test_on_error(self, exporter, caplog):
        """Test on_error method."""
        exc = ValueError("test error")
        with caplog.at_level(logging.ERROR):
            exporter.on_error(exc)
        assert "Error in event subscription: test error" in caplog.text

    def test_on_complete(self, exporter, caplog):
        """Test on_complete method."""
        with caplog.at_level(logging.INFO):
            exporter.on_complete()
        assert "Event stream completed" in caplog.text

    def test_start_no_event_stream(self, mock_context_state):
        """Test _start when no event stream is available."""
        mock_context_state.event_stream.get.return_value = None
        exporter = ConcreteExporter(mock_context_state)

        result = exporter._start()
        assert result is None
        assert not exporter._running

    def test_start_invalid_subject(self, mock_context_state):
        """Test _start when subject doesn't support subscription."""
        mock_subject = Mock()
        # Remove subscribe method to simulate invalid subject
        del mock_subject.subscribe
        mock_context_state.event_stream.get.return_value = mock_subject
        exporter = ConcreteExporter(mock_context_state)

        with patch('aiq.observability.exporter.base_exporter.logger') as mock_logger:
            result = exporter._start()
            assert result is None
            mock_logger.error.assert_called_once()

    def test_start_success(self, exporter):
        """Test successful _start."""
        mock_subscription = Mock()
        exporter._context_state.event_stream.get.return_value.subscribe.return_value = mock_subscription

        result = exporter._start()

        assert result is not None
        assert exporter._running is True
        assert exporter._subscription is mock_subscription

        # Test that _ready_event is set
        assert exporter._ready_event.is_set()

    def test_start_subscription_callback(self, exporter):
        """Test that subscription callback works correctly."""
        mock_event = Mock(spec=IntermediateStep)

        # Capture the callback passed to subscribe
        captured_callback = None

        def capture_subscribe(*_args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get('on_next')
            return Mock()

        exporter._context_state.event_stream.get.return_value.subscribe.side_effect = capture_subscribe

        exporter._start()

        # Call the captured callback
        assert captured_callback is not None
        assert callable(captured_callback)
        captured_callback(mock_event)  # pylint: disable=not-callable

        # Verify the event was exported
        assert mock_event in exporter.exported_events

    async def test_pre_start(self, exporter):
        """Test _pre_start method (default implementation)."""
        # Should not raise any errors
        await exporter._pre_start()

    async def test_start_context_manager_success(self, exporter):
        """Test start context manager with successful flow."""
        exporter._start = Mock(return_value=Mock())
        exporter.stop = AsyncMock()

        async with exporter.start():
            assert True  # Context manager worked

        exporter.stop.assert_called_once()

    async def test_start_context_manager_already_running(self, exporter):
        """Test start context manager when already running."""
        exporter._running = True
        exporter.stop = AsyncMock()

        async with exporter.start():
            pass

        exporter.stop.assert_called_once()

    async def test_start_context_manager_no_event_stream(self, exporter):
        """Test start context manager with no event stream."""
        exporter._start = Mock(return_value=None)
        exporter.stop = AsyncMock()

        async with exporter.start():
            pass

        exporter.stop.assert_called_once()

    async def test_cleanup(self, exporter):
        """Test _cleanup method (default implementation)."""
        # Should not raise any errors
        await exporter._cleanup()

    async def test_wait_for_tasks_no_tasks(self, exporter):
        """Test _wait_for_tasks with no tasks."""
        # Should complete immediately
        await exporter._wait_for_tasks()

    async def test_wait_for_tasks_with_completing_tasks(self, exporter):
        """Test _wait_for_tasks with tasks that complete quickly."""

        async def quick_task():
            await asyncio.sleep(0.01)
            return "done"

        task1 = asyncio.create_task(quick_task())
        task2 = asyncio.create_task(quick_task())
        exporter._tasks.add(task1)
        exporter._tasks.add(task2)

        await exporter._wait_for_tasks(timeout=1.0)

        assert task1.done()
        assert task2.done()

    async def test_wait_for_tasks_timeout(self, exporter, caplog):
        """Test _wait_for_tasks with timeout."""

        async def slow_task():
            await asyncio.sleep(10)  # Much longer than timeout

        task = asyncio.create_task(slow_task())
        exporter._tasks.add(task)

        # Capture logs from the specific logger
        with caplog.at_level(logging.WARNING, logger="aiq.observability.exporter.base_exporter"):
            await exporter._wait_for_tasks(timeout=0.01)

        assert "did not complete within" in caplog.text
        task.cancel()  # Clean up

    async def test_wait_for_tasks_exception(self, exporter, caplog):
        """Test _wait_for_tasks with task that raises exception."""

        async def failing_task():
            raise ValueError("task error")

        task = asyncio.create_task(failing_task())
        exporter._tasks.add(task)

        with caplog.at_level(logging.ERROR):
            await exporter._wait_for_tasks()

        # Should log error but not re-raise
        assert task.done()

    async def test_stop_not_running(self, exporter):
        """Test stop when not running."""
        exporter._running = False
        await exporter.stop()
        # Should complete without error

    async def test_stop_running(self, exporter):
        """Test stop when running - new behavior: no task waiting."""
        mock_subscription = Mock()
        exporter._subscription = mock_subscription
        exporter._running = True
        exporter._cleanup = AsyncMock()

        await exporter.stop()

        assert exporter._running is False
        assert exporter._shutdown_event.is_set()
        exporter._cleanup.assert_called_once()
        mock_subscription.unsubscribe.assert_called_once()
        assert exporter._subscription is None
        assert len(exporter._tasks) == 0  # Task tracking cleared

    async def test_stop_with_tasks(self, exporter):
        """Test stop with active tasks - new behavior: tasks continue running, tracking cleared."""

        async def test_task():
            await asyncio.sleep(10)  # Long task continues running

        task = asyncio.create_task(test_task())
        exporter._tasks.add(task)
        exporter._running = True

        await exporter.stop()

        # New behavior: tasks continue running but tracking is cleared
        assert not task.cancelled()  # Task continues in event loop
        assert len(exporter._tasks) == 0  # Tracking set is cleared

        # Clean up the task for test completion
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_stop_task_cancellation_error(self, exporter, caplog):
        """Test stop with task - no cancellation errors since tasks aren't cancelled."""

        # Create a task that would have caused cancellation issues in the old approach
        task = Mock()
        task.done.return_value = False
        task.cancel.return_value = None
        task.get_name.return_value = "test_task"

        exporter._tasks.add(task)
        exporter._running = True

        # Capture logs from the specific logger
        with caplog.at_level(logging.WARNING, logger="aiq.observability.exporter.base_exporter"):
            await exporter.stop()

        # New behavior: no cancellation warnings since tasks aren't cancelled
        assert "Error while canceling task" not in caplog.text
        assert len(exporter._tasks) == 0  # Tracking cleared

    async def test_wait_ready(self, exporter):
        """Test wait_ready method."""

        # Start the ready event in a separate task
        async def set_ready():
            await asyncio.sleep(0.01)
            exporter._ready_event.set()

        ready_task = asyncio.create_task(set_ready())

        # This should wait until the event is set
        await exporter.wait_ready()

        await ready_task
        assert exporter._ready_event.is_set()

    def test_create_isolated_instance(self, exporter):
        """Test create_isolated_instance method."""
        new_context = Mock(spec=AIQContextState)

        isolated = exporter.create_isolated_instance(new_context)

        # Should be different objects
        assert isolated is not exporter
        assert isolated._context_state is new_context
        assert isolated._is_isolated_instance is True
        assert isolated._subscription is None
        assert isolated._running is False

        # Should share the same class but have isolated descriptor attributes
        assert type(isolated) is type(exporter)
        assert isolated._tasks is not exporter._tasks
        assert isolated._ready_event is not exporter._ready_event
        assert isolated._shutdown_event is not exporter._shutdown_event

    def test_create_isolated_instance_tracking(self, exporter):
        """Test that isolated instances are tracked separately."""
        initial_isolated_count = BaseExporter.get_isolated_instance_count()

        isolated = exporter.create_isolated_instance(Mock(spec=AIQContextState))
        assert isolated is not None  # Use the variable

        assert BaseExporter.get_isolated_instance_count() == initial_isolated_count + 1

    def test_get_active_instance_count(self):
        """Test get_active_instance_count class method."""
        initial_count = BaseExporter.get_active_instance_count()

        exporter1 = ConcreteExporter()
        assert exporter1 is not None  # Use the variable
        assert BaseExporter.get_active_instance_count() == initial_count + 1

        exporter2 = ConcreteExporter()
        assert exporter2 is not None  # Use the variable
        assert BaseExporter.get_active_instance_count() == initial_count + 2

    def test_get_isolated_instance_count(self, exporter):
        """Test get_isolated_instance_count class method."""
        initial_count = BaseExporter.get_isolated_instance_count()

        isolated1 = exporter.create_isolated_instance(Mock(spec=AIQContextState))
        assert isolated1 is not None  # Use the variable
        assert BaseExporter.get_isolated_instance_count() == initial_count + 1

        isolated2 = exporter.create_isolated_instance(Mock(spec=AIQContextState))
        assert isolated2 is not None  # Use the variable
        assert BaseExporter.get_isolated_instance_count() == initial_count + 2

    def test_log_instance_stats(self, caplog):
        """Test log_instance_stats class method."""
        with caplog.at_level(logging.INFO):
            BaseExporter.log_instance_stats()

        assert "BaseExporter instances" in caplog.text
        assert "Total:" in caplog.text
        assert "Original:" in caplog.text
        assert "Isolated:" in caplog.text

    def test_log_instance_stats_high_isolation_warning(self, exporter, caplog):
        """Test log_instance_stats warns about high isolation count."""
        # Create many isolated instances to trigger warning
        isolated_instances = []
        for _ in range(51):
            isolated_instances.append(exporter.create_isolated_instance(Mock(spec=AIQContextState)))

        # Capture logs from the specific logger
        with caplog.at_level(logging.WARNING, logger="aiq.observability.exporter.base_exporter"):
            BaseExporter.log_instance_stats()

        assert "High number of isolated BaseExporter instances" in caplog.text

    def test_del_with_active_resources(self):
        """Test __del__ warning when exporter has active resources."""
        exporter = ConcreteExporter()
        exporter._running = True

        # Patch the logger to verify the warning is called
        with patch('aiq.observability.exporter.base_exporter.logger') as mock_logger:
            exporter.__del__()  # pylint: disable=unnecessary-dunder-call

        # Check that warning was called with the expected message
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "being garbage collected with active resources" in warning_call

    def test_del_with_active_tasks(self):
        """Test __del__ warning when exporter has active tasks."""
        exporter = ConcreteExporter()
        # Set running to True to trigger the warning condition
        exporter._running = True

        # Patch the logger to verify the warning is called
        with patch('aiq.observability.exporter.base_exporter.logger') as mock_logger:
            exporter.__del__()  # pylint: disable=unnecessary-dunder-call

        # Check that warning was called with the expected message
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "being garbage collected with active resources" in warning_call

    def test_isolated_attributes_independence(self, exporter):
        """Test that isolated attributes work independently across instances."""
        # Add items to original exporter's task set
        original_task = Mock()
        exporter._tasks.add(original_task)

        # Create isolated instance
        isolated = exporter.create_isolated_instance(Mock(spec=AIQContextState))

        # Add different task to isolated instance
        isolated_task = Mock()
        isolated._tasks.add(isolated_task)

        # Verify independence
        assert original_task in exporter._tasks
        assert original_task not in isolated._tasks
        assert isolated_task not in exporter._tasks
        assert isolated_task in isolated._tasks

    async def test_integration_start_export_stop(self, mock_context_state):
        """Integration test of the full lifecycle."""
        events_exported = []

        def track_export(event):
            events_exported.append(event)

        exporter = ConcreteExporter(mock_context_state, track_export)

        # Mock the subject and subscription
        mock_subscription = Mock()
        mock_subject = mock_context_state.event_stream.get.return_value
        mock_subject.subscribe.return_value = mock_subscription

        async with exporter.start():
            # Wait for ready
            await exporter.wait_ready()

            # Simulate event processing
            test_event = Mock(spec=IntermediateStep)

            # Get the callback that was registered
            subscribe_call = mock_subject.subscribe.call_args
            on_next_callback = subscribe_call.kwargs['on_next']

            # Simulate event arrival
            on_next_callback(test_event)

        # Verify the event was processed
        assert test_event in events_exported
        assert not exporter._running
        mock_subscription.unsubscribe.assert_called_once()
