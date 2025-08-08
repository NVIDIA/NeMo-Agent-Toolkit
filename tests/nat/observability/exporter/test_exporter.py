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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.observability.exporter.exporter import Exporter


class TestExporter:
    """Test cases for the abstract Exporter class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that the abstract Exporter class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class Exporter"):
            Exporter()  # pylint: disable=abstract-class-instantiated

    def test_abstract_methods_exist(self):
        """Test that all expected abstract methods are defined."""
        abstract_methods = Exporter.__abstractmethods__
        expected_methods = {'start', 'stop', 'export', 'on_error', 'on_complete'}
        assert abstract_methods == expected_methods

    def test_concrete_implementation_requires_all_methods(self):
        """Test that a concrete implementation must implement all abstract methods."""

        # Missing one method should fail
        class IncompleteExporter(Exporter):

            async def start(self) -> AsyncGenerator[None]:
                yield

            async def stop(self) -> None:
                pass

            def export(self, event: IntermediateStep) -> None:
                pass

            def on_error(self, exc: Exception) -> None:
                pass

            # Missing on_complete

        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteExporter"):
            IncompleteExporter()  # pylint: disable=abstract-class-instantiated


class ConcreteExporter(Exporter):
    """Concrete implementation of Exporter for testing purposes."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.exported_events = []
        self.errors = []
        self.completed = False

    @asynccontextmanager
    async def start(self) -> AsyncGenerator[None]:
        """Start the exporter and yield control."""
        self.started = True
        try:
            yield
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the exporter."""
        self.stopped = True

    def export(self, event: IntermediateStep) -> None:
        """Export an event."""
        self.exported_events.append(event)

    def on_error(self, exc: Exception) -> None:
        """Handle an error."""
        self.errors.append(exc)

    def on_complete(self) -> None:
        """Handle completion."""
        self.completed = True


class TestConcreteExporter:
    """Test cases for a concrete implementation of Exporter."""

    @pytest.fixture
    def exporter(self):
        """Create a concrete exporter instance for testing."""
        return ConcreteExporter()

    @pytest.fixture
    def mock_intermediate_step(self):
        """Create a mock IntermediateStep for testing."""
        return Mock(spec=IntermediateStep)

    def test_concrete_implementation_can_be_instantiated(self, exporter):
        """Test that a concrete implementation can be instantiated."""
        assert isinstance(exporter, Exporter)
        assert isinstance(exporter, ConcreteExporter)

    async def test_start_stop_lifecycle(self, exporter):
        """Test the start/stop lifecycle of the exporter."""
        assert not exporter.started
        assert not exporter.stopped

        async with exporter.start():
            assert exporter.started
            assert not exporter.stopped

        assert exporter.stopped

    async def test_start_context_manager_behavior(self, exporter):
        """Test that start() works as an async context manager."""
        async with exporter.start():
            # Inside context, should be started but not stopped
            assert exporter.started
            assert not exporter.stopped

        # Outside context, should be stopped
        assert exporter.stopped

    async def test_start_handles_exceptions(self, exporter):
        """Test that start() properly handles exceptions and still calls stop()."""
        with pytest.raises(ValueError):
            async with exporter.start():
                assert exporter.started
                raise ValueError("Test exception")

        # Should still be stopped even when exception occurred
        assert exporter.stopped

    def test_export_functionality(self, exporter, mock_intermediate_step):
        """Test the export functionality."""
        assert len(exporter.exported_events) == 0

        exporter.export(mock_intermediate_step)

        assert len(exporter.exported_events) == 1
        assert exporter.exported_events[0] is mock_intermediate_step

    def test_export_multiple_events(self, exporter):
        """Test exporting multiple events."""
        events = [Mock(spec=IntermediateStep) for _ in range(3)]

        for event in events:
            exporter.export(event)

        assert len(exporter.exported_events) == 3
        assert exporter.exported_events == events

    def test_on_error_functionality(self, exporter):
        """Test the error handling functionality."""
        assert len(exporter.errors) == 0

        test_exception = ValueError("Test error")
        exporter.on_error(test_exception)

        assert len(exporter.errors) == 1
        assert exporter.errors[0] is test_exception

    def test_on_error_multiple_errors(self, exporter):
        """Test handling multiple errors."""
        errors = [ValueError("Error 1"), RuntimeError("Error 2"), Exception("Error 3")]

        for error in errors:
            exporter.on_error(error)

        assert len(exporter.errors) == 3
        assert exporter.errors == errors

    def test_on_complete_functionality(self, exporter):
        """Test the completion handling functionality."""
        assert not exporter.completed

        exporter.on_complete()

        assert exporter.completed

    def test_on_complete_idempotent(self, exporter):
        """Test that on_complete can be called multiple times safely."""
        exporter.on_complete()
        assert exporter.completed

        # Should not raise an error if called again
        exporter.on_complete()
        assert exporter.completed

    async def test_full_workflow_integration(self, exporter):
        """Test a complete workflow with start, export, error, complete, and stop."""
        test_event = Mock(spec=IntermediateStep)
        test_error = RuntimeError("Workflow error")

        async with exporter.start():
            # Export an event
            exporter.export(test_event)
            assert len(exporter.exported_events) == 1
            assert exporter.exported_events[0] is test_event

            # Handle an error
            exporter.on_error(test_error)
            assert len(exporter.errors) == 1
            assert exporter.errors[0] is test_error

            # Complete the workflow
            exporter.on_complete()
            assert exporter.completed

        # Verify final state
        assert exporter.started
        assert exporter.stopped
        assert exporter.completed
        assert len(exporter.exported_events) == 1
        assert len(exporter.errors) == 1

    def test_initial_state(self, exporter):
        """Test that the exporter starts in the correct initial state."""
        assert not exporter.started
        assert not exporter.stopped
        assert not exporter.completed
        assert len(exporter.exported_events) == 0
        assert len(exporter.errors) == 0
