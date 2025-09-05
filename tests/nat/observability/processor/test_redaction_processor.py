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
from unittest.mock import patch

import pytest

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.observability.processor.redaction_processor import RedactionProcessor
from nat.observability.processor.redaction_processor import SpanRedactionProcessor

logger = logging.getLogger(__name__)


# Concrete test implementations
class ConcreteRedactionProcessor(RedactionProcessor[str]):
    """Concrete implementation of RedactionProcessor for testing string redaction."""

    def __init__(self, should_redact_result: bool = True, redacted_value: str = "[REDACTED]"):
        self.should_redact_result = should_redact_result
        self.redacted_value = redacted_value
        self.should_redact_called = False
        self.redact_item_called = False
        self.should_redact_calls = []
        self.redact_item_calls = []

    def should_redact(self, item: str, context: Context) -> bool:
        """Test implementation that tracks calls and returns configured result."""
        self.should_redact_called = True
        self.should_redact_calls.append((item, context))
        return self.should_redact_result

    def redact_item(self, item: str) -> str:
        """Test implementation that replaces content with redacted value."""
        self.redact_item_called = True
        self.redact_item_calls.append(item)
        return self.redacted_value


class ErroringRedactionProcessor(RedactionProcessor[str]):
    """Redaction processor that raises errors for testing error handling."""

    def __init__(self, should_redact_error: bool = False, redact_item_error: bool = False):
        self.should_redact_error = should_redact_error
        self.redact_item_error = redact_item_error

    def should_redact(self, item: str, context: Context) -> bool:
        """Raises error if configured to do so."""
        if self.should_redact_error:
            raise RuntimeError("should_redact failed")
        return True

    def redact_item(self, item: str) -> str:
        """Raises error if configured to do so."""
        if self.redact_item_error:
            raise RuntimeError("redact_item failed")
        return "[REDACTED]"


class ConcreteSpanRedactionProcessor(SpanRedactionProcessor):
    """Concrete implementation of SpanRedactionProcessor for testing span redaction."""

    def __init__(self, should_redact_result: bool = True, redact_span_name: bool = True):
        self.should_redact_result = should_redact_result
        self.redact_span_name = redact_span_name
        self.should_redact_called = False
        self.redact_item_called = False
        self.should_redact_calls = []
        self.redact_item_calls = []

    def should_redact(self, item: Span, context: Context) -> bool:
        """Test implementation for span redaction check."""
        self.should_redact_called = True
        self.should_redact_calls.append((item, context))
        return self.should_redact_result

    def redact_item(self, item: Span) -> Span:
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

            class IncompleteProcessor(RedactionProcessor[str]):
                pass

            IncompleteProcessor()  # type: ignore

        # Missing redact_item method
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class MissingRedactItem(RedactionProcessor[str]):

                def should_redact(self, item: str, context: Context) -> bool:
                    return True

            MissingRedactItem()  # type: ignore

        # Missing should_redact method
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class MissingShouldRedact(RedactionProcessor[str]):

                def redact_item(self, item: str) -> str:
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

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_with_redaction_enabled(self, mock_context_get, mock_context):
        """Test process method when should_redact returns True."""
        mock_context_get.return_value = mock_context

        processor = ConcreteRedactionProcessor(should_redact_result=True, redacted_value="SAFE_VALUE")
        input_item = "sensitive_data"

        result = await processor.process(input_item)

        assert result == "SAFE_VALUE"
        assert processor.should_redact_called
        assert processor.redact_item_called
        assert len(processor.should_redact_calls) == 1
        assert processor.should_redact_calls[0] == (input_item, mock_context)
        assert len(processor.redact_item_calls) == 1
        assert processor.redact_item_calls[0] == input_item

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_with_redaction_disabled(self, mock_context_get, mock_context):
        """Test process method when should_redact returns False."""
        mock_context_get.return_value = mock_context

        processor = ConcreteRedactionProcessor(should_redact_result=False)
        input_item = "normal_data"

        result = await processor.process(input_item)

        assert result == input_item  # Should return original item unchanged
        assert processor.should_redact_called
        assert not processor.redact_item_called  # Should not redact
        assert len(processor.should_redact_calls) == 1
        assert processor.should_redact_calls[0] == (input_item, mock_context)
        assert len(processor.redact_item_calls) == 0

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_context_retrieval(self, mock_context_get):
        """Test that process method properly retrieves context."""
        mock_context = Mock(spec=Context)
        mock_context_get.return_value = mock_context

        processor = ConcreteRedactionProcessor(should_redact_result=False)
        input_item = "test_data"

        await processor.process(input_item)

        mock_context_get.assert_called_once()
        assert processor.should_redact_calls[0][1] is mock_context

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_multiple_items(self, mock_context_get, mock_context):
        """Test processing multiple items maintains state correctly."""
        mock_context_get.return_value = mock_context

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
        assert processor.should_redact_calls[0][0] == "item1"
        assert processor.should_redact_calls[1][0] == "item2"
        assert processor.should_redact_calls[2][0] == "item3"


class TestRedactionProcessorErrorHandling:
    """Test error handling in RedactionProcessor."""

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_should_redact_error_propagates(self, mock_context_get, mock_context):
        """Test that errors in should_redact are propagated."""
        mock_context_get.return_value = mock_context

        processor = ErroringRedactionProcessor(should_redact_error=True)

        with pytest.raises(RuntimeError, match="should_redact failed"):
            await processor.process("test_item")

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_redact_item_error_propagates(self, mock_context_get, mock_context):
        """Test that errors in redact_item are propagated."""
        mock_context_get.return_value = mock_context

        processor = ErroringRedactionProcessor(redact_item_error=True)

        with pytest.raises(RuntimeError, match="redact_item failed"):
            await processor.process("test_item")

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_process_context_get_error_propagates(self, mock_context_get):
        """Test that errors in Context.get() are propagated."""
        mock_context_get.side_effect = RuntimeError("Context retrieval failed")

        processor = ConcreteRedactionProcessor()

        with pytest.raises(RuntimeError, match="Context retrieval failed"):
            await processor.process("test_item")


class TestSpanRedactionProcessor:
    """Test SpanRedactionProcessor class."""

    def test_span_redaction_processor_is_abstract(self):
        """Test that SpanRedactionProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SpanRedactionProcessor()  # type: ignore

    def test_span_redaction_processor_inheritance(self):
        """Test that SpanRedactionProcessor properly inherits from RedactionProcessor."""
        processor = ConcreteSpanRedactionProcessor()
        assert isinstance(processor, SpanRedactionProcessor)
        assert isinstance(processor, RedactionProcessor)
        assert hasattr(processor, 'should_redact')
        assert hasattr(processor, 'redact_item')
        assert hasattr(processor, 'process')

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_span_redaction_processor_redacts_span(self, mock_context_get, mock_context, sample_span):
        """Test that SpanRedactionProcessor can redact span data."""
        mock_context_get.return_value = mock_context

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

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_span_redaction_processor_no_redaction(self, mock_context_get, mock_context, sample_span):
        """Test that SpanRedactionProcessor passes through spans when not redacting."""
        mock_context_get.return_value = mock_context

        processor = ConcreteSpanRedactionProcessor(should_redact_result=False)

        result = await processor.process(sample_span)

        assert result is sample_span  # Should return exact same object
        assert processor.should_redact_called
        assert not processor.redact_item_called


class TestRedactionProcessorTypeHandling:
    """Test type handling in RedactionProcessor."""

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_different_input_types(self, mock_context_get, mock_context):
        """Test redaction processor with different input types."""
        mock_context_get.return_value = mock_context

        # Test with integer input
        class IntRedactionProcessor(RedactionProcessor[int]):

            def should_redact(self, item: int, context: Context) -> bool:
                return item > 100  # Redact large numbers

            def redact_item(self, item: int) -> int:
                return 0  # Redact to zero

        processor = IntRedactionProcessor()

        # Test with small number (no redaction)
        result1 = await processor.process(50)
        assert result1 == 50

        # Test with large number (redaction)
        result2 = await processor.process(200)
        assert result2 == 0

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_dict_redaction_processor(self, mock_context_get, mock_context):
        """Test redaction processor with dictionary input."""
        mock_context_get.return_value = mock_context

        class DictRedactionProcessor(RedactionProcessor[dict]):

            def should_redact(self, item: dict, context: Context) -> bool:
                return any("sensitive" in key for key in item.keys())

            def redact_item(self, item: dict) -> dict:
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

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_redaction_with_none_input(self, mock_context_get, mock_context):
        """Test redaction processor behavior with None input."""
        mock_context_get.return_value = mock_context

        class NullableRedactionProcessor(RedactionProcessor[str | None]):

            def should_redact(self, item: str | None, context: Context) -> bool:
                return item is not None and "sensitive" in item

            def redact_item(self, item: str | None) -> str | None:
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

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_redaction_preserves_object_identity_when_not_redacting(self, mock_context_get, mock_context):
        """Test that original object is returned when not redacting."""
        mock_context_get.return_value = mock_context

        processor = ConcreteRedactionProcessor(should_redact_result=False)
        input_item = "test_string"

        result = await processor.process(input_item)

        assert result is input_item  # Should be exact same object reference

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_redaction_state_isolation(self, mock_context_get, mock_context):
        """Test that processor state is properly isolated between calls."""
        mock_context_get.return_value = mock_context

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
        assert processor1.should_redact_calls[0][0] == "item1"
        assert processor2.should_redact_calls[0][0] == "item2"


class TestRedactionProcessorTypeIntrospection:
    """Test type introspection capabilities."""

    def test_string_redaction_processor_types(self):
        """Test type introspection for string redaction processor."""
        processor = ConcreteRedactionProcessor()

        assert processor.input_type is str
        assert processor.output_type is str
        assert processor.input_class is str
        assert processor.output_class is str

    def test_span_redaction_processor_types(self):
        """Test type introspection for span redaction processor."""
        processor = ConcreteSpanRedactionProcessor()

        assert processor.input_type is Span
        assert processor.output_type is Span
        assert processor.input_class is Span
        assert processor.output_class is Span


class TestRedactionProcessorLogging:
    """Test logging behavior in RedactionProcessor."""

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_no_default_logging_in_process_method(self, mock_context_get, mock_context, caplog):
        """Test that process method doesn't log by default."""
        mock_context_get.return_value = mock_context

        processor = ConcreteRedactionProcessor(should_redact_result=True)

        with caplog.at_level(logging.DEBUG):
            await processor.process("test_item")

        # The base process method should not log anything by default
        # Logging would be implemented in concrete should_redact/redact_item methods
        # Filter out any logs that come from other parts of the system
        redaction_logs = [record for record in caplog.records if 'redaction_processor' in record.name]
        assert len(redaction_logs) == 0

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_custom_logging_in_concrete_implementations(self, mock_context_get, mock_context, caplog):
        """Test that concrete implementations can add their own logging."""
        mock_context_get.return_value = mock_context

        class LoggingRedactionProcessor(RedactionProcessor[str]):

            def should_redact(self, item: str, context: Context) -> bool:
                logger.info("Checking if item should be redacted: %s", item)
                return "sensitive" in item

            def redact_item(self, item: str) -> str:
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

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_span_redaction_integration(self, mock_context_get, mock_context, sample_span):
        """Test full span redaction integration."""
        mock_context_get.return_value = mock_context

        # Create a processor that redacts spans with "sensitive" in the name
        class SensitiveSpanRedactionProcessor(SpanRedactionProcessor):

            def should_redact(self, item: Span, context: Context) -> bool:
                return "sensitive" in item.name.lower()

            def redact_item(self, item: Span) -> Span:
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

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_conditional_redaction_based_on_context(self, mock_context_get, sample_span):
        """Test redaction decisions based on context."""

        # Create a processor that only redacts in production context
        class ContextAwareRedactionProcessor(SpanRedactionProcessor):

            def should_redact(self, item: Span, context: Context) -> bool:
                # Mock context has environment attribute
                return getattr(context, 'environment', 'dev') == 'production'

            def redact_item(self, item: Span) -> Span:
                return Span(name="[REDACTED]",
                            context=item.context,
                            parent=item.parent,
                            start_time=item.start_time,
                            end_time=item.end_time,
                            status=item.status,
                            attributes=item.attributes,
                            events=item.events)

        processor = ContextAwareRedactionProcessor()

        # Test with dev context (no redaction)
        dev_context = Mock(spec=Context)
        dev_context.environment = 'dev'
        mock_context_get.return_value = dev_context

        result1 = await processor.process(sample_span)
        assert result1 is sample_span

        # Test with production context (redaction)
        prod_context = Mock(spec=Context)
        prod_context.environment = 'production'
        mock_context_get.return_value = prod_context

        result2 = await processor.process(sample_span)
        assert result2.name == "[REDACTED]"
        # Verify context ID is preserved (if context exists)
        if sample_span.context and result2.context:
            assert result2.context.span_id == sample_span.context.span_id


class TestRedactionProcessorPerformance:
    """Test performance-related aspects of RedactionProcessor."""

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_efficient_no_redaction_path(self, mock_context_get, mock_context):
        """Test that no-redaction path is efficient (no unnecessary object creation)."""
        mock_context_get.return_value = mock_context

        class EfficientProcessor(RedactionProcessor[dict]):

            def should_redact(self, item: dict, context: Context) -> bool:
                return False  # Never redact

            def redact_item(self, item: dict) -> dict:
                # This should never be called
                raise AssertionError("redact_item should not be called when should_redact returns False")

        processor = EfficientProcessor()
        input_dict = {"key": "value"}

        result = await processor.process(input_dict)

        # Should return exact same object reference (no copying)
        assert result is input_dict

    @patch('nat.observability.processor.redaction_processor.Context.get')
    async def test_context_retrieval_called_once_per_process(self, mock_context_get, mock_context):
        """Test that Context.get() is called exactly once per process() call."""
        mock_context_get.return_value = mock_context

        processor = ConcreteRedactionProcessor(should_redact_result=True)

        await processor.process("test_item")

        # Verify Context.get() was called exactly once
        mock_context_get.assert_called_once()

        # Reset mock for second call
        mock_context_get.reset_mock()

        await processor.process("another_item")

        # Should be called again for the second process call
        mock_context_get.assert_called_once()
