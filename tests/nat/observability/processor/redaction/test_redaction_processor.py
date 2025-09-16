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

import logging
from unittest.mock import Mock

import pytest

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.redaction.redaction_processor import RedactionContext
from nat.observability.processor.redaction.redaction_processor import RedactionContextState
from nat.observability.processor.redaction.redaction_processor import RedactionManager
from nat.observability.processor.redaction.redaction_processor import RedactionProcessor

logger = logging.getLogger(__name__)


# Concrete test implementations
class ConcreteRedactionProcessor(RedactionProcessor[str, str]):
    """Concrete implementation of RedactionProcessor for testing string redaction."""

    def __init__(self, should_redact_result: bool = True, redacted_value: str = "[REDACTED]"):
        self.should_redact_result = should_redact_result
        self.redacted_value = redacted_value
        self.should_redact_called = False
        self.redact_item_called = False
        self.should_redact_calls = []
        self.redact_item_calls = []

    async def should_redact(self, item: str) -> bool:
        """Test implementation that tracks calls and returns configured result."""
        self.should_redact_called = True
        self.should_redact_calls.append(item)
        return self.should_redact_result

    async def redact_item(self, item: str) -> str:
        """Test implementation that replaces content with redacted value."""
        self.redact_item_called = True
        self.redact_item_calls.append(item)
        return self.redacted_value


class ErroringRedactionProcessor(RedactionProcessor[str, str]):
    """Redaction processor that raises errors for testing error handling."""

    def __init__(self, should_redact_error: bool = False, redact_item_error: bool = False):
        self.should_redact_error = should_redact_error
        self.redact_item_error = redact_item_error

    async def should_redact(self, item: str) -> bool:
        """Raises error if configured to do so."""
        if self.should_redact_error:
            raise RuntimeError("should_redact failed")
        return True

    async def redact_item(self, item: str) -> str:
        """Raises error if configured to do so."""
        if self.redact_item_error:
            raise RuntimeError("redact_item failed")
        return "[REDACTED]"


class ConcreteSpanRedactionProcessor(RedactionProcessor[Span, Span]):
    """Concrete implementation of RedactionProcessor for testing span redaction."""

    def __init__(self, should_redact_result: bool = True, redact_span_name: bool = True):
        self.should_redact_result = should_redact_result
        self.redact_span_name = redact_span_name
        self.should_redact_called = False
        self.redact_item_called = False
        self.should_redact_calls = []
        self.redact_item_calls = []

    async def should_redact(self, item: Span) -> bool:
        """Test implementation for span redaction check."""
        self.should_redact_called = True
        self.should_redact_calls.append(item)
        return self.should_redact_result

    async def redact_item(self, item: Span) -> Span:
        """Test implementation that redacts span name."""
        self.redact_item_called = True
        self.redact_item_calls.append(item)

        if self.redact_span_name:
            # Create a copy with redacted name
            redacted_span = Span(name="[REDACTED]",
                                 context=item.context,
                                 parent=item.parent,
                                 start_time=item.start_time,
                                 end_time=item.end_time,
                                 status=item.status,
                                 attributes=item.attributes,
                                 events=item.events)
            return redacted_span
        return item


@pytest.fixture
def mock_context():
    """Create a mock context."""
    return Mock(spec=Context)


@pytest.fixture
def sample_span():
    """Create a sample span for testing."""
    span_context = SpanContext(
        span_id=123,  # Using int as per the model
        trace_id=456)
    return Span(name="sensitive_operation",
                context=span_context,
                parent=None,
                start_time=1000000,
                end_time=2000000,
                attributes={"key": "value"},
                events=[])


class TestRedactionProcessorAbstractBehavior:
    """Test abstract behavior of RedactionProcessor."""

    def test_redaction_processor_is_abstract(self):
        """Test that RedactionProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RedactionProcessor()  # type: ignore

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations cannot be instantiated."""

        # Missing both abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteProcessor(RedactionProcessor[str, str]):
                pass

            IncompleteProcessor()  # type: ignore

        # Missing redact_item method
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class MissingRedactItem(RedactionProcessor[str, str]):

                async def should_redact(self, item: str) -> bool:
                    return True

            MissingRedactItem()  # type: ignore

        # Missing should_redact method
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class MissingShouldRedact(RedactionProcessor[str, str]):

                async def redact_item(self, item: str) -> str:
                    return "[REDACTED]"

            MissingShouldRedact()  # type: ignore

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        processor = ConcreteRedactionProcessor()
        assert isinstance(processor, RedactionProcessor)
        assert hasattr(processor, 'should_redact')
        assert hasattr(processor, 'redact_item')
        assert hasattr(processor, 'process')


class TestRedactionProcessorProcess:
    """Test the process method of RedactionProcessor."""

    async def test_process_with_redaction_enabled(self):
        """Test process method when should_redact returns True."""
        processor = ConcreteRedactionProcessor(should_redact_result=True, redacted_value="SAFE_VALUE")
        input_item = "sensitive_data"

        result = await processor.process(input_item)

        assert result == "SAFE_VALUE"
        assert processor.should_redact_called
        assert processor.redact_item_called
        assert len(processor.should_redact_calls) == 1
        assert processor.should_redact_calls[0] == input_item
        assert len(processor.redact_item_calls) == 1
        assert processor.redact_item_calls[0] == input_item

    async def test_process_with_redaction_disabled(self):
        """Test process method when should_redact returns False."""
        processor = ConcreteRedactionProcessor(should_redact_result=False)
        input_item = "normal_data"

        result = await processor.process(input_item)

        assert result == input_item  # Should return original item unchanged
        assert processor.should_redact_called
        assert not processor.redact_item_called  # Should not redact
        assert len(processor.should_redact_calls) == 1
        assert processor.should_redact_calls[0] == input_item
        assert len(processor.redact_item_calls) == 0

    async def test_process_multiple_items(self):
        """Test processing multiple items maintains state correctly."""
        processor = ConcreteRedactionProcessor(should_redact_result=True, redacted_value="[HIDDEN]")

        # Process multiple items
        result1 = await processor.process("item1")
        result2 = await processor.process("item2")
        result3 = await processor.process("item3")

        assert result1 == "[HIDDEN]"
        assert result2 == "[HIDDEN]"
        assert result3 == "[HIDDEN]"

        # Verify all calls were tracked
        assert len(processor.should_redact_calls) == 3
        assert len(processor.redact_item_calls) == 3
        assert processor.should_redact_calls[0] == "item1"
        assert processor.should_redact_calls[1] == "item2"
        assert processor.should_redact_calls[2] == "item3"


class TestRedactionProcessorErrorHandling:
    """Test error handling in RedactionProcessor."""

    async def test_process_should_redact_error_propagates(self):
        """Test that errors in should_redact are propagated."""
        processor = ErroringRedactionProcessor(should_redact_error=True)

        with pytest.raises(RuntimeError, match="should_redact failed"):
            await processor.process("test_item")

    async def test_process_redact_item_error_propagates(self):
        """Test that errors in redact_item are propagated."""
        processor = ErroringRedactionProcessor(redact_item_error=True)

        with pytest.raises(RuntimeError, match="redact_item failed"):
            await processor.process("test_item")


class TestSpanRedactionProcessor:
    """Test RedactionProcessor with Span types."""

    def test_span_redaction_processor_inheritance(self):
        """Test that ConcreteSpanRedactionProcessor properly inherits from RedactionProcessor."""
        processor = ConcreteSpanRedactionProcessor()
        assert isinstance(processor, RedactionProcessor)
        assert hasattr(processor, 'should_redact')
        assert hasattr(processor, 'redact_item')
        assert hasattr(processor, 'process')

    async def test_span_redaction_processor_redacts_span(self, sample_span):
        """Test that SpanRedactionProcessor can redact span data."""
        processor = ConcreteSpanRedactionProcessor(should_redact_result=True, redact_span_name=True)

        result = await processor.process(sample_span)

        assert result.name == "[REDACTED]"
        # Verify the context IDs are preserved (if context exists)
        if sample_span.context and result.context:
            assert result.context.span_id == sample_span.context.span_id
            assert result.context.trace_id == sample_span.context.trace_id
        assert result.attributes == sample_span.attributes
        assert processor.should_redact_called
        assert processor.redact_item_called

    async def test_span_redaction_processor_no_redaction(self, sample_span):
        """Test that SpanRedactionProcessor passes through spans when not redacting."""
        processor = ConcreteSpanRedactionProcessor(should_redact_result=False)

        result = await processor.process(sample_span)

        assert result is sample_span  # Should return exact same object
        assert processor.should_redact_called
        assert not processor.redact_item_called


class TestRedactionProcessorTypeHandling:
    """Test type handling in RedactionProcessor."""

    async def test_different_input_types(self):
        """Test redaction processor with different input types."""

        # Test with integer input
        class IntRedactionProcessor(RedactionProcessor[int, int]):

            async def should_redact(self, item: int) -> bool:
                return item > 100  # Redact large numbers

            async def redact_item(self, item: int) -> int:
                return 0  # Redact to zero

        processor = IntRedactionProcessor()

        # Test with small number (no redaction)
        result1 = await processor.process(50)
        assert result1 == 50

        # Test with large number (redaction)
        result2 = await processor.process(200)
        assert result2 == 0

    async def test_dict_redaction_processor(self):
        """Test redaction processor with dictionary input."""

        class DictRedactionProcessor(RedactionProcessor[dict, dict]):

            async def should_redact(self, item: dict) -> bool:
                return any("sensitive" in key for key in item.keys())

            async def redact_item(self, item: dict) -> dict:
                # Return new dict with sensitive keys redacted
                return {k: "[REDACTED]" if "sensitive" in k else v for k, v in item.items()}

        processor = DictRedactionProcessor()

        # Test with non-sensitive data
        safe_data = {"name": "John", "age": 30}
        result1 = await processor.process(safe_data)
        assert result1 == safe_data

        # Test with sensitive data
        sensitive_data = {"name": "John", "sensitive_field": "secret", "age": 30}
        result2 = await processor.process(sensitive_data)
        assert result2 == {"name": "John", "sensitive_field": "[REDACTED]", "age": 30}


class TestRedactionProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_redaction_with_none_input(self):
        """Test redaction processor behavior with None input."""

        class NullableRedactionProcessor(RedactionProcessor[str | None, str]):

            async def should_redact(self, item: str | None) -> bool:
                return item is not None and "sensitive" in item

            async def redact_item(self, item: str | None) -> str | None:
                if item is None:
                    return None
                return "[REDACTED]"

        processor = NullableRedactionProcessor()

        # Test with None input
        result1 = await processor.process(None)
        assert result1 is None

        # Test with non-sensitive string
        result2 = await processor.process("normal_data")
        assert result2 == "normal_data"

        # Test with sensitive string
        result3 = await processor.process("sensitive_data")
        assert result3 == "[REDACTED]"

    async def test_redaction_preserves_object_identity_when_not_redacting(self):
        """Test that original object is returned when not redacting."""

        processor = ConcreteRedactionProcessor(should_redact_result=False)
        input_item = "test_string"

        result = await processor.process(input_item)

        assert result is input_item  # Should be exact same object reference

    async def test_redaction_state_isolation(self):
        """Test that processor state is properly isolated between calls."""

        processor1 = ConcreteRedactionProcessor(should_redact_result=True)
        processor2 = ConcreteRedactionProcessor(should_redact_result=False)

        # Process with both processors
        await processor1.process("item1")
        await processor2.process("item2")

        # Verify state isolation
        assert processor1.should_redact_called
        assert processor1.redact_item_called
        assert processor2.should_redact_called
        assert not processor2.redact_item_called

        assert len(processor1.should_redact_calls) == 1
        assert len(processor2.should_redact_calls) == 1
        assert processor1.should_redact_calls[0] == "item1"
        assert processor2.should_redact_calls[0] == "item2"


class TestRedactionProcessorTypeIntrospection:
    """Test type introspection capabilities."""

    def test_string_redaction_processor_types(self):
        """Test type introspection for string redaction processor."""
        processor = ConcreteRedactionProcessor()

        assert processor.input_type is str
        assert processor.output_type is str

        # Test Pydantic-based validation methods (preferred approach)
        assert processor.validate_input_type("test_string")
        assert not processor.validate_input_type(123)
        assert processor.validate_output_type("result_string")
        assert not processor.validate_output_type(123)

    def test_span_redaction_processor_types(self):
        """Test type introspection for span redaction processor."""
        processor = ConcreteSpanRedactionProcessor()

        assert processor.input_type is Span
        assert processor.output_type is Span

        # Test Pydantic-based validation methods (preferred approach)
        test_span = Span(name="test", span_id="123", trace_id="456")
        assert processor.validate_input_type(test_span)
        assert not processor.validate_input_type("not_a_span")
        assert processor.validate_output_type(test_span)


class TestRedactionProcessorLogging:
    """Test logging behavior in RedactionProcessor."""

    async def test_no_default_logging_in_process_method(self, caplog):
        """Test that process method doesn't log by default."""

        processor = ConcreteRedactionProcessor(should_redact_result=True)

        with caplog.at_level(logging.DEBUG):
            await processor.process("test_item")

        # The base process method should not log anything by default
        # Logging would be implemented in concrete should_redact/redact_item methods
        # Filter out any logs that come from other parts of the system
        redaction_logs = [record for record in caplog.records if 'redaction_processor' in record.name]
        assert len(redaction_logs) == 0

    async def test_custom_logging_in_concrete_implementations(self, caplog):
        """Test that concrete implementations can add their own logging."""

        class LoggingRedactionProcessor(RedactionProcessor[str, str]):

            async def should_redact(self, item: str) -> bool:
                logger.info("Checking if item should be redacted: %s", item)
                return "sensitive" in item

            async def redact_item(self, item: str) -> str:
                logger.info("Redacting item: %s", item)
                return "[REDACTED]"

        processor = LoggingRedactionProcessor()

        with caplog.at_level(logging.INFO):
            await processor.process("sensitive_data")

        # Should see logs from our concrete implementation
        assert "Checking if item should be redacted: sensitive_data" in caplog.text
        assert "Redacting item: sensitive_data" in caplog.text


class TestRedactionProcessorIntegration:
    """Test integration scenarios with RedactionProcessor."""

    async def test_span_redaction_integration(self, sample_span):
        """Test full span redaction integration."""

        # Create a processor that redacts spans with "sensitive" in the name
        class SensitiveSpanRedactionProcessor(RedactionProcessor[Span, Span]):

            async def should_redact(self, item: Span) -> bool:
                return "sensitive" in item.name.lower()

            async def redact_item(self, item: Span) -> Span:
                return Span(name="[OPERATION_REDACTED]",
                            context=item.context,
                            parent=item.parent,
                            start_time=item.start_time,
                            end_time=item.end_time,
                            status=item.status,
                            attributes={
                                k: "[REDACTED]" if "password" in k.lower() else v
                                for k, v in item.attributes.items()
                            },
                            events=item.events)

        processor = SensitiveSpanRedactionProcessor()

        # Test with sensitive span name
        result = await processor.process(sample_span)

        assert result.name == "[OPERATION_REDACTED]"
        # Verify context IDs are preserved (if context exists)
        if sample_span.context and result.context:
            assert result.context.span_id == sample_span.context.span_id
            assert result.context.trace_id == sample_span.context.trace_id
        assert result.attributes == {"key": "value"}  # No password attribute, so unchanged

    async def test_conditional_redaction_based_on_context(self, sample_span):
        """Test redaction decisions based on context."""

        # Create a processor that only redacts in production context
        class ContextAwareRedactionProcessor(RedactionProcessor[Span, Span]):

            async def should_redact(self, item: Span) -> bool:
                # For this test, we'll simulate environment-based redaction differently
                # Since we don't have context parameter, we'll use a simple rule
                return "production" in item.name.lower()

            async def redact_item(self, item: Span) -> Span:
                return Span(name="[REDACTED]",
                            context=item.context,
                            parent=item.parent,
                            start_time=item.start_time,
                            end_time=item.end_time,
                            status=item.status,
                            attributes=item.attributes,
                            events=item.events)

        processor = ContextAwareRedactionProcessor()

        # Test with non-production span (no redaction)
        result1 = await processor.process(sample_span)
        assert result1 is sample_span

        # Test with production-related span (redaction)
        production_span = Span(name="production_operation",
                               context=sample_span.context,
                               parent=sample_span.parent,
                               start_time=sample_span.start_time,
                               end_time=sample_span.end_time,
                               status=sample_span.status,
                               attributes=sample_span.attributes,
                               events=sample_span.events)

        result2 = await processor.process(production_span)
        assert result2.name == "[REDACTED]"
        # Verify context ID is preserved (if context exists)
        if production_span.context and result2.context:
            assert result2.context.span_id == production_span.context.span_id


class TestRedactionProcessorPerformance:
    """Test performance-related aspects of RedactionProcessor."""

    async def test_efficient_no_redaction_path(self):
        """Test that no-redaction path is efficient (no unnecessary object creation)."""

        class EfficientProcessor(RedactionProcessor[dict, dict]):

            async def should_redact(self, item: dict) -> bool:
                return False  # Never redact

            async def redact_item(self, item: dict) -> dict:
                # This should never be called
                raise AssertionError("redact_item should not be called when should_redact returns False")

        processor = EfficientProcessor()
        input_dict = {"key": "value"}

        result = await processor.process(input_dict)

        # Should return exact same object reference (no copying)
        assert result is input_dict

    async def test_multiple_process_calls_work_correctly(self):
        """Test that multiple process() calls work correctly."""
        processor = ConcreteRedactionProcessor(should_redact_result=True)

        await processor.process("test_item")

        # Verify processor works correctly

        # Test second call
        await processor.process("another_item")


# =============================================================================
# RedactionContextState Tests
# =============================================================================


class TestRedactionContextState:
    """Test RedactionContextState class."""

    def test_redaction_context_state_initialization(self):
        """Test that RedactionContextState initializes correctly."""
        state = RedactionContextState()

        assert hasattr(state, 'redaction_result')
        assert state.redaction_result is not None
        assert state.redaction_result.get() is None

    def test_redaction_context_state_default_factory(self):
        """Test that the default factory creates a ContextVar with correct default."""
        state = RedactionContextState()

        # Should start with None value
        assert state.redaction_result.get() is None

        # Should be able to set and get values
        state.redaction_result.set(True)
        assert state.redaction_result.get() is True

        state.redaction_result.set(False)
        assert state.redaction_result.get() is False

    def test_multiple_redaction_context_states_are_independent(self):
        """Test that multiple RedactionContextState instances are independent."""
        state1 = RedactionContextState()
        state2 = RedactionContextState()

        # Set different values
        state1.redaction_result.set(True)
        state2.redaction_result.set(False)

        # Values should be independent
        assert state1.redaction_result.get() is True
        assert state2.redaction_result.get() is False

    def test_redaction_context_state_reset_to_none(self):
        """Test that RedactionContextState can be reset to None."""
        state = RedactionContextState()

        # Set a value
        state.redaction_result.set(True)
        assert state.redaction_result.get() is True

        # Reset to None
        state.redaction_result.set(None)
        assert state.redaction_result.get() is None


# =============================================================================
# RedactionManager Tests
# =============================================================================


class TestRedactionManager:
    """Test RedactionManager class."""

    @pytest.fixture
    def context_state(self):
        """Create a RedactionContextState for testing."""
        return RedactionContextState()

    @pytest.fixture
    def manager(self, context_state):
        """Create a RedactionManager for testing."""
        return RedactionManager(context_state)

    def test_redaction_manager_initialization(self, context_state):
        """Test that RedactionManager initializes correctly."""
        manager = RedactionManager(context_state)

        assert manager._context_state is context_state

    def test_set_redaction_result_true(self, manager, context_state):
        """Test setting redaction result to True."""
        manager.set_redaction_result(True)

        assert context_state.redaction_result.get() is True

    def test_set_redaction_result_false(self, manager, context_state):
        """Test setting redaction result to False."""
        manager.set_redaction_result(False)

        assert context_state.redaction_result.get() is False

    def test_clear_redaction_result(self, manager, context_state):
        """Test clearing redaction result."""
        # Set a value first
        manager.set_redaction_result(True)
        assert context_state.redaction_result.get() is True

        # Clear it
        manager.clear_redaction_result()
        assert context_state.redaction_result.get() is None

    async def test_redaction_check_with_sync_function(self, manager):
        """Test redaction_check with a synchronous function."""

        def sync_callback(data):
            return data == "sensitive"

        # Test with sensitive data
        result = await manager.redaction_check(sync_callback, "sensitive")
        assert result is True

        # Clear cache and test with non-sensitive data
        manager.clear_redaction_result()
        result = await manager.redaction_check(sync_callback, "normal")
        assert result is False

    async def test_redaction_check_with_async_function(self, manager):
        """Test redaction_check with an asynchronous function."""

        async def async_callback(data):
            return len(data) > 5

        # Test with long data
        result = await manager.redaction_check(async_callback, "very_long_string")
        assert result is True

        # Clear cache and test with short data
        manager.clear_redaction_result()
        result = await manager.redaction_check(async_callback, "short")
        assert result is False

    async def test_redaction_check_caching(self, manager, context_state):
        """Test that redaction_check caches results within the same context."""
        call_count = 0

        def counting_callback(_data):
            nonlocal call_count
            call_count += 1
            return True

        # First call should execute callback
        result1 = await manager.redaction_check(counting_callback, "test_data")
        assert result1 is True
        assert call_count == 1

        # Second call should use cached result
        result2 = await manager.redaction_check(counting_callback, "different_data")
        assert result2 is True
        assert call_count == 1  # Should not increment

        # Verify cached value is set
        assert context_state.redaction_result.get() is True

    async def test_redaction_check_with_falsy_return_value(self, manager):
        """Test redaction_check properly handles falsy return values."""

        def falsy_callback(data):
            return 0  # Falsy but not None

        result = await manager.redaction_check(falsy_callback, "test")
        assert result is False

    async def test_redaction_check_with_truthy_return_value(self, manager):
        """Test redaction_check properly handles truthy return values."""

        def truthy_callback(data):
            return "non_empty_string"  # Truthy

        result = await manager.redaction_check(truthy_callback, "test")
        assert result is True

    async def test_redaction_check_with_generator(self, manager):
        """Test redaction_check with a generator function."""

        def generator_callback(data):
            yield "processing"
            return data == "sensitive"  # This is the return value that ainvoke_any will use

        result = await manager.redaction_check(generator_callback, "sensitive")
        assert result is True

    async def test_redaction_check_with_async_generator(self, manager):
        """Test redaction_check with an async generator function."""

        async def async_generator_callback(data):
            yield len(data) > 3

        result = await manager.redaction_check(async_generator_callback, "long_data")
        assert result is True

    async def test_redaction_check_cache_clear_and_reset(self, manager, context_state):
        """Test that clearing cache allows new callback execution."""
        call_count = 0

        def counting_callback(data):
            nonlocal call_count
            call_count += 1
            return call_count % 2 == 1  # Alternates between True/False

        # First call
        result1 = await manager.redaction_check(counting_callback, "test1")
        assert result1 is True
        assert call_count == 1

        # Clear cache
        manager.clear_redaction_result()

        # Second call should execute callback again
        result2 = await manager.redaction_check(counting_callback, "test2")
        assert result2 is False
        assert call_count == 2

    async def test_redaction_check_error_propagation(self, manager):
        """Test that errors in callbacks are properly propagated."""

        def error_callback(_data):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await manager.redaction_check(error_callback, "test_data")


# =============================================================================
# RedactionContext Tests
# =============================================================================


class TestRedactionContext:
    """Test RedactionContext class."""

    @pytest.fixture
    def context_state(self):
        """Create a RedactionContextState for testing."""
        return RedactionContextState()

    @pytest.fixture
    def redaction_context(self, context_state):
        """Create a RedactionContext for testing."""
        return RedactionContext(context_state)

    def test_redaction_context_initialization(self, context_state):
        """Test that RedactionContext initializes correctly."""
        context = RedactionContext(context_state)

        assert context._context_state is context_state

    def test_redaction_result_property_none(self, redaction_context):
        """Test redaction_result property when no result is set."""
        result = redaction_context.redaction_result
        assert result is None

    def test_redaction_result_property_true(self, redaction_context, context_state):
        """Test redaction_result property when result is True."""
        context_state.redaction_result.set(True)

        result = redaction_context.redaction_result
        assert result is True

    def test_redaction_result_property_false(self, redaction_context, context_state):
        """Test redaction_result property when result is False."""
        context_state.redaction_result.set(False)

        result = redaction_context.redaction_result
        assert result is False

    async def test_redaction_manager_context_manager(self, redaction_context):
        """Test that redaction_manager returns a proper context manager."""
        async with redaction_context.redaction_manager() as manager:
            assert isinstance(manager, RedactionManager)
            assert manager._context_state is redaction_context._context_state

    async def test_redaction_manager_context_manager_functionality(self, redaction_context):
        """Test full functionality through the context manager."""
        async with redaction_context.redaction_manager() as manager:
            # Test setting result
            manager.set_redaction_result(True)
            assert redaction_context.redaction_result is True

            # Test callback execution
            def test_callback(data):
                return data == "test"

            result = await manager.redaction_check(test_callback, "test")
            assert result is True

            # Verify caching
            assert redaction_context.redaction_result is True

    async def test_multiple_context_managers(self, redaction_context):
        """Test that multiple context managers share the same state."""
        # Set initial state
        redaction_context._context_state.redaction_result.set(True)

        async with redaction_context.redaction_manager() as manager1:
            async with redaction_context.redaction_manager() as manager2:
                # Both managers should see the same state
                assert manager1._context_state is manager2._context_state

                # Changes through one manager should be visible through the other
                manager1.set_redaction_result(False)
                assert redaction_context.redaction_result is False

                manager2.set_redaction_result(True)
                assert redaction_context.redaction_result is True

    async def test_redaction_context_isolation(self):
        """Test that different RedactionContext instances are isolated."""
        state1 = RedactionContextState()
        state2 = RedactionContextState()

        context1 = RedactionContext(state1)
        context2 = RedactionContext(state2)

        # Set different values
        state1.redaction_result.set(True)
        state2.redaction_result.set(False)

        # Contexts should return different values
        assert context1.redaction_result is True
        assert context2.redaction_result is False

        # Context managers should be independent
        async with context1.redaction_manager() as manager1:
            async with context2.redaction_manager() as manager2:
                manager1.set_redaction_result(False)
                manager2.set_redaction_result(True)

                assert context1.redaction_result is False
                assert context2.redaction_result is True


# =============================================================================
# Integration Tests for All Classes
# =============================================================================


class TestRedactionComponentsIntegration:
    """Test integration between RedactionContextState, RedactionManager, and RedactionContext."""

    async def test_full_redaction_workflow(self):
        """Test complete redaction workflow using all components."""
        # Create context state
        state = RedactionContextState()
        context = RedactionContext(state)

        # Define a callback that determines redaction
        def should_redact_callback(data):
            return "sensitive" in data.lower()

        # Test through context manager
        async with context.redaction_manager() as manager:
            # Test non-sensitive data
            result1 = await manager.redaction_check(should_redact_callback, "normal_data")
            assert result1 is False
            assert context.redaction_result is False

            # Clear and test sensitive data
            manager.clear_redaction_result()
            result2 = await manager.redaction_check(should_redact_callback, "sensitive_information")
            assert result2 is True
            assert context.redaction_result is True

    async def test_context_state_persistence_across_managers(self):
        """Test that context state persists across different manager instances."""
        state = RedactionContextState()
        context = RedactionContext(state)

        # Set value through first manager
        async with context.redaction_manager() as manager1:
            manager1.set_redaction_result(True)

        # Verify value persists through second manager
        async with context.redaction_manager() as manager2:
            assert context.redaction_result is True

            # Test callback caching across managers
            call_count = 0

            def counting_callback(_data):
                nonlocal call_count
                call_count += 1
                return False

            # Should use cached result, not call callback
            result = await manager2.redaction_check(counting_callback, "test")
            assert result is True  # Uses cached True value
            assert call_count == 0  # Callback not called

    async def test_redaction_state_lifecycle(self):
        """Test the complete lifecycle of redaction state."""
        state = RedactionContextState()
        context = RedactionContext(state)

        # Initial state
        assert context.redaction_result is None

        async with context.redaction_manager() as manager:
            # Set initial result
            manager.set_redaction_result(True)
            assert context.redaction_result is True

            # Test callback with cached result
            def never_called_callback(_data):
                raise AssertionError("Should not be called due to caching")

            result = await manager.redaction_check(never_called_callback, "any_data")
            assert result is True

            # Clear and verify reset
            manager.clear_redaction_result()
            assert context.redaction_result is None

            # Now callback should be called
            def actual_callback(data):
                return data == "test"

            result = await manager.redaction_check(actual_callback, "test")
            assert result is True
            assert context.redaction_result is True
