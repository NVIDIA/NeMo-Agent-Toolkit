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
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode
from nat.observability.exporter.raw_exporter import RawExporter
from nat.observability.processor.processor import Processor
from nat.utils.reactive.subject import Subject


class MockProcessor(Processor[IntermediateStep, str]):
    """Mock processor for testing."""

    def __init__(self, name: str = "MockProcessor", should_fail: bool = False):
        super().__init__()
        self.name = name
        self.should_fail = should_fail
        self.process_called = False
        self.processed_items = []

    async def process(self, item: IntermediateStep) -> str:
        self.process_called = True
        self.processed_items.append(item)
        if self.should_fail:
            raise RuntimeError(f"Processor {self.name} failed")
        return f"processed_{item.UUID}"


class StringProcessor(Processor[str, str]):
    """Mock processor that processes strings to strings."""

    def __init__(self, name: str = "StringProcessor", should_fail: bool = False):
        super().__init__()
        self.name = name
        self.should_fail = should_fail
        self.process_called = False
        self.processed_items = []

    async def process(self, item: str) -> str:
        self.process_called = True
        self.processed_items.append(item)
        if self.should_fail:
            raise RuntimeError(f"Processor {self.name} failed")
        return f"string_processed_{item}"


class ConcreteRawExporter(RawExporter[IntermediateStep, str]):
    """Concrete implementation of RawExporter for testing."""

    def __init__(self, context_state: ContextState | None = None):
        super().__init__(context_state)
        self.exported_items = []
        self.export_processed_called = False

    async def export_processed(self, item: str) -> None:
        """Mock implementation that records exported items."""
        self.export_processed_called = True
        self.exported_items.append(item)


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
def raw_exporter(mock_context_state):
    """Create a concrete raw exporter for testing."""
    return ConcreteRawExporter(mock_context_state)


@pytest.fixture
def sample_intermediate_step():
    """Create a sample IntermediateStep for testing."""
    payload = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                      name="test_tool",
                                      tags=["test"],
                                      UUID="test-uuid-123")
    return IntermediateStep(parent_id="root",
                            function_ancestry=InvocationNode(function_name="test_tool", function_id="test-function-id"),
                            payload=payload)


class TestRawExporterCleanMocking:
    """Tests using clean mocking strategies without warnings."""

    def test_export_type_checking(self, raw_exporter, sample_intermediate_step):
        """Test export type checking without async complications."""
        # Strategy 1: Test the type checking logic directly

        # Valid input should pass the isinstance check
        with patch.object(raw_exporter, '_create_export_task') as mock_create_task:
            raw_exporter.export(sample_intermediate_step)
            mock_create_task.assert_called_once()

            # Clean up any created coroutines
            args = mock_create_task.call_args[0]
            if args and hasattr(args[0], 'close'):
                args[0].close()

        # Invalid inputs should not call _create_export_task
        invalid_inputs = [None, "string", 123, [], {}, Mock()]

        with patch.object(raw_exporter, '_create_export_task') as mock_create_task:
            for invalid_input in invalid_inputs:
                raw_exporter.export(invalid_input)

            mock_create_task.assert_not_called()

    def test_export_method_signature_and_behavior(self, raw_exporter):
        """Test that export method has correct signature and behavior."""
        # Strategy 2: Test method signature and basic behavior
        import inspect

        # Check method signature
        sig = inspect.signature(raw_exporter.export)
        params = list(sig.parameters.keys())
        assert len(params) == 1
        assert params[0] == 'event'

        # Test method exists and is callable
        assert hasattr(raw_exporter, 'export')
        assert callable(raw_exporter.export)

    async def test_processing_pipeline_directly(self, raw_exporter, sample_intermediate_step):
        """Test the processing pipeline by calling it directly."""
        # Strategy 3: Test async methods directly without complex mocking
        processor = MockProcessor("test_processor")
        raw_exporter.add_processor(processor)

        # Call the async method directly
        await raw_exporter._export_with_processing(sample_intermediate_step)

        # Verify results
        assert processor.process_called
        assert len(processor.processed_items) == 1
        assert processor.processed_items[0] is sample_intermediate_step
        assert raw_exporter.export_processed_called
        assert raw_exporter.exported_items[0] == f"processed_{sample_intermediate_step.UUID}"

    def test_export_with_proper_async_mock(self, raw_exporter, sample_intermediate_step):
        """Test export using proper async mocking that doesn't create warnings."""
        # Strategy 4: Simple mocking without task creation

        with patch.object(raw_exporter, '_create_export_task') as mock_create_task:
            # Mock to just clean up the coroutine
            def cleanup_coro(coro):
                if hasattr(coro, 'close'):
                    coro.close()
                return Mock()  # Return a mock task

            mock_create_task.side_effect = cleanup_coro

            raw_exporter.export(sample_intermediate_step)

            mock_create_task.assert_called_once()


class TestRawExporterCoreLogic:
    """Test core logic without complex async mocking."""

    def test_inheritance_and_abstract_methods(self):
        """Test inheritance structure and abstract method enforcement."""
        # Test that RawExporter is abstract
        from abc import ABC
        assert issubclass(RawExporter, ABC)

        # Test that incomplete implementations fail
        class IncompleteExporter(RawExporter[IntermediateStep, str]):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteExporter()  # type: ignore[misc]

    def test_initialization_patterns(self, mock_context_state):
        """Test different initialization patterns."""
        # With context state
        exporter1 = ConcreteRawExporter(mock_context_state)
        assert exporter1._context_state is mock_context_state

        # Without context state (uses default)
        with patch('nat.builder.context.ContextState.get') as mock_get:
            mock_get.return_value = mock_context_state
            exporter2 = ConcreteRawExporter()
            assert exporter2._context_state is mock_context_state
            mock_get.assert_called_once()

    async def test_processor_integration(self, raw_exporter, sample_intermediate_step):
        """Test processor integration without export method complications."""
        # Test with single processor
        processor1 = MockProcessor("proc1")
        raw_exporter.add_processor(processor1)

        await raw_exporter._export_with_processing(sample_intermediate_step)

        assert processor1.process_called
        assert raw_exporter.export_processed_called

        # Test with multiple processors - use compatible types
        raw_exporter.exported_items.clear()
        raw_exporter.export_processed_called = False

        # Clear existing processors and add a chain: IntermediateStep -> str -> str
        raw_exporter.clear_processors()
        processor_step_to_str = MockProcessor("step_to_str")
        processor_str_to_str = StringProcessor("str_to_str")

        raw_exporter.add_processor(processor_step_to_str)
        raw_exporter.add_processor(processor_str_to_str)

        await raw_exporter._export_with_processing(sample_intermediate_step)

        assert processor_step_to_str.process_called
        assert processor_str_to_str.process_called
        assert raw_exporter.export_processed_called

    async def test_error_handling(self, raw_exporter, sample_intermediate_step, caplog):
        """Test error handling in processing pipeline."""
        failing_processor = MockProcessor("failing_proc", should_fail=True)
        raw_exporter.add_processor(failing_processor)

        with pytest.raises(ValueError, match="is not a valid output type"):
            with caplog.at_level(logging.ERROR):
                await raw_exporter._export_with_processing(sample_intermediate_step)

        assert failing_processor.process_called
        assert "Error in processor" in caplog.text


class TestRawExporterMinimalMocking:
    """Tests using minimal mocking for maximum clarity."""

    def test_export_behavioral_contract(self, raw_exporter):
        """Test the behavioral contract of export method."""
        # The export method should:
        # 1. Only accept IntermediateStep objects
        # 2. Call _create_export_task for valid inputs
        # 3. Do nothing for invalid inputs

        call_count = 0

        def counting_create_task(coro):
            nonlocal call_count
            call_count += 1
            # Clean up coroutine immediately
            if hasattr(coro, 'close'):
                coro.close()

        with patch.object(raw_exporter, '_create_export_task', side_effect=counting_create_task):
            # Valid input
            payload = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                              name="test",
                                              tags=[],
                                              UUID="test-123")
            valid_step = IntermediateStep(parent_id="root",
                                          function_ancestry=InvocationNode(function_name="test",
                                                                           function_id="test-function-id"),
                                          payload=payload)
            raw_exporter.export(valid_step)

            # Invalid inputs
            raw_exporter.export(None)
            raw_exporter.export("string")
            raw_exporter.export(123)
            raw_exporter.export([])

            # Should only be called once for the valid input
            assert call_count == 1

    def test_processing_chain_logic(self, mock_context_state):
        """Test processing chain logic with concrete implementations."""

        class TestExporter(RawExporter[IntermediateStep, str]):

            def __init__(self):
                super().__init__(mock_context_state)
                self.results = []

            async def export_processed(self, item: str):
                self.results.append(item)

        exporter = TestExporter()

        # Test with processor
        processor = MockProcessor("converter")
        exporter.add_processor(processor)

        payload = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                          name="test",
                                          tags=[],
                                          UUID="no-proc-123")
        step = IntermediateStep(parent_id="root",
                                function_ancestry=InvocationNode(function_name="test", function_id="test-function-id"),
                                payload=payload)

        asyncio.run(exporter._export_with_processing(step))

        assert len(exporter.results) == 1
        assert exporter.results[0] == "processed_no-proc-123"

    def test_integration_with_real_async_execution(self, mock_context_state):
        """Test integration using real async execution."""

        class AsyncTestExporter(RawExporter[IntermediateStep, str]):

            def __init__(self):
                super().__init__(mock_context_state)
                self.exported_items = []
                self.tasks_created = []

            async def export_processed(self, item: str):
                self.exported_items.append(item)

            def _create_export_task(self, coro):
                # Store the coroutine for later execution instead of creating task immediately
                self.tasks_created.append(coro)

        exporter = AsyncTestExporter()
        processor = MockProcessor("real_processor")
        exporter.add_processor(processor)

        # Create test data
        payload = IntermediateStepPayload(event_type=IntermediateStepType.WORKFLOW_END,
                                          name="integration_test",
                                          tags=["integration"],
                                          UUID="real-async-123")
        step = IntermediateStep(parent_id="root",
                                function_ancestry=InvocationNode(function_name="integration_test",
                                                                 function_id="test-function-id"),
                                payload=payload)

        # Call export (stores coroutine)
        exporter.export(step)

        # Execute the coroutine manually
        async def execute_stored_coroutines():
            for coro in exporter.tasks_created:
                await coro

        asyncio.run(execute_stored_coroutines())

        # Verify results
        assert len(exporter.tasks_created) == 1
        assert processor.process_called
        assert len(exporter.exported_items) == 1
        assert exporter.exported_items[0] == "processed_real-async-123"


class TestRawExporterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_export_with_none_and_falsy_values(self, raw_exporter):
        """Test export with various falsy values."""
        falsy_values = [None, False, 0, "", [], {}]

        with patch.object(raw_exporter, '_create_export_task') as mock_create_task:
            for falsy_value in falsy_values:
                raw_exporter.export(falsy_value)

            mock_create_task.assert_not_called()

    def test_type_checking_precision(self, raw_exporter):
        """Test that type checking is precise, not just truthy."""

        # Create objects that might fool weak type checking
        class FakeIntermediateStep:

            def __init__(self):
                self.UUID = "fake-uuid"
                self.payload = Mock()

        fake_step = FakeIntermediateStep()

        with patch.object(raw_exporter, '_create_export_task') as mock_create_task:
            raw_exporter.export(fake_step)
            mock_create_task.assert_not_called()

    async def test_processor_edge_cases(self, mock_context_state):
        """Test processor edge cases."""

        class EdgeCaseExporter(RawExporter[IntermediateStep, str]):

            def __init__(self):
                super().__init__(mock_context_state)
                self.results = []

            async def export_processed(self, item: str):
                self.results.append(item)

        exporter = EdgeCaseExporter()

        # Test with processor that returns empty string
        class EmptyProcessor(Processor[IntermediateStep, str]):

            async def process(self, item: IntermediateStep) -> str:
                return ""

        exporter.add_processor(EmptyProcessor())

        payload = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                          name="edge_test",
                                          tags=[],
                                          UUID="edge-123")
        step = IntermediateStep(parent_id="root",
                                function_ancestry=InvocationNode(function_name="edge_test",
                                                                 function_id="test-function-id"),
                                payload=payload)

        await exporter._export_with_processing(step)

        assert len(exporter.results) == 1
        assert exporter.results[0] == ""
