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

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import BaseModel

from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.plugins.data_flywheel.observability.processor.dfw_record_processor import DFWToDictProcessor
from nat.plugins.data_flywheel.observability.processor.dfw_record_processor import SpanToDFWRecordProcessor


class MockDFWRecord(BaseModel):
    """Mock DFW record for testing purposes."""

    record_id: str
    name: str
    value: int = 42
    optional_field: str | None = None


class MockTargetRecord(BaseModel):
    """Mock target record type for SpanToDFWRecordProcessor testing."""

    target_id: str
    converted_data: Any
    source: str = "span"


class TestDFWToDictProcessor:
    """Test suite for DFWToDictProcessor class."""

    def test_processor_inheritance(self):
        """Test that DFWToDictProcessor properly inherits from Processor."""
        processor = DFWToDictProcessor()

        # Should have type introspection capabilities
        assert hasattr(processor, 'input_type')
        assert hasattr(processor, 'output_type')

        # Input type should be generic (bound to BaseModel)
        assert processor.output_type is dict

    async def test_process_valid_dfw_record(self):
        """Test processing a valid DFW record to dictionary."""
        processor = DFWToDictProcessor()

        # Create a mock DFW record
        record = MockDFWRecord(record_id="test-123", name="Test Record", value=100, optional_field="optional_value")

        result = await processor.process(record)

        # Should return dictionary with expected fields
        expected = {"record_id": "test-123", "name": "Test Record", "value": 100, "optional_field": "optional_value"}
        assert result == expected
        assert isinstance(result, dict)

    async def test_process_record_with_aliases(self):
        """Test that field aliases are properly handled."""
        processor = DFWToDictProcessor()

        record = MockDFWRecord(record_id="alias-test", name="Alias Test")
        result = await processor.process(record)

        # Should have record_id field
        assert "record_id" in result
        assert result["record_id"] == "alias-test"

    async def test_process_record_with_none_values(self):
        """Test processing record with None values."""
        processor = DFWToDictProcessor()

        record = MockDFWRecord(record_id="none-test", name="None Test", optional_field=None)
        result = await processor.process(record)

        assert result["optional_field"] is None
        assert "record_id" in result

    async def test_process_none_item_returns_empty_dict(self):
        """Test that None input returns empty dictionary."""
        processor = DFWToDictProcessor()

        result = await processor.process(None)

        assert result == {}
        assert isinstance(result, dict)

    async def test_process_preserves_nested_structures(self):
        """Test that nested data structures are preserved in the output."""

        class NestedDFWRecord(BaseModel):
            record_id: str
            nested_dict: dict[str, Any]
            nested_list: list[int]

        processor = DFWToDictProcessor()

        nested_data = {"key1": "value1", "key2": {"nested": "data"}, "key3": [1, 2, 3]}

        record = NestedDFWRecord(record_id="nested-test", nested_dict=nested_data, nested_list=[10, 20, 30])

        result = await processor.process(record)

        assert result["nested_dict"] == nested_data
        assert result["nested_list"] == [10, 20, 30]

    async def test_model_dump_json_called_correctly(self):
        """Test that model_dump_json is called with correct parameters."""
        processor = DFWToDictProcessor()

        # Create a mock record
        record = MagicMock(spec=BaseModel)
        record.model_dump_json.return_value = '{"test": "value"}'

        result = await processor.process(record)

        # Verify model_dump_json was called with by_alias=True
        record.model_dump_json.assert_called_once_with(by_alias=True)
        assert result == {"test": "value"}


class TestSpanToDFWRecordProcessor:
    """Test suite for SpanToDFWRecordProcessor class."""

    def test_processor_initialization(self):
        """Test processor initialization with client_id."""
        client_id = "test-client-123"
        processor = SpanToDFWRecordProcessor(client_id=client_id)

        assert processor._client_id == client_id

        # Should have type introspection capabilities
        assert hasattr(processor, 'input_type')
        assert hasattr(processor, 'output_type')

    def test_processor_inheritance(self):
        """Test that SpanToDFWRecordProcessor properly inherits from Processor and TypeIntrospectionMixin."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        # Should be a Processor
        assert hasattr(processor, 'process')

        # Should have TypeIntrospectionMixin capabilities
        assert hasattr(processor, 'input_type')
        assert hasattr(processor, 'output_type')

    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.span_to_dfw_record')
    async def test_process_llm_start_event(self, mock_span_to_dfw_record):
        """Test processing span with LLM_START event type."""
        # Setup processor
        client_id = "test-client"
        processor = SpanToDFWRecordProcessor(client_id=client_id)

        # Mock the conversion function
        mock_converted_record = MockTargetRecord(target_id="converted-123", converted_data="test")
        mock_span_to_dfw_record.return_value = mock_converted_record

        # Create test span
        span = Span(name="test-llm-span",
                    context=SpanContext(),
                    attributes={"nat.event_type": IntermediateStepType.LLM_START})

        result = await processor.process(span)

        # Verify span_to_dfw_record was called correctly
        mock_span_to_dfw_record.assert_called_once_with(span=span, to_type=processor.output_type, client_id=client_id)

        assert result == mock_converted_record

    async def test_process_unsupported_event_type_returns_none(self):
        """Test that unsupported event types return None."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        # Create span with unsupported event type
        span = Span(
            name="test-span",
            context=SpanContext(),
            attributes={"nat.event_type": IntermediateStepType.TOOL_START}  # Not LLM_START
        )

        result = await processor.process(span)

        assert result is None

    async def test_process_span_without_event_type_returns_none(self):
        """Test processing span without nat.event_type attribute."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        span = Span(
            name="test-span",
            context=SpanContext(),
            attributes={}  # No event_type
        )

        result = await processor.process(span)

        assert result is None

    async def test_process_span_with_none_event_type_returns_none(self):
        """Test processing span with None as event_type."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        span = Span(name="test-span", context=SpanContext(), attributes={"nat.event_type": None})

        result = await processor.process(span)

        assert result is None

    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.span_to_dfw_record')
    async def test_process_span_to_dfw_record_returns_none(self, mock_span_to_dfw_record):
        """Test handling when span_to_dfw_record returns None."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        # Mock conversion function to return None
        mock_span_to_dfw_record.return_value = None

        span = Span(name="test-span",
                    context=SpanContext(),
                    attributes={"nat.event_type": IntermediateStepType.LLM_START})

        result = await processor.process(span)

        assert result is None
        mock_span_to_dfw_record.assert_called_once()

    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.span_to_dfw_record')
    async def test_process_passes_correct_parameters(self, mock_span_to_dfw_record):
        """Test that all parameters are passed correctly to span_to_dfw_record."""
        client_id = "specific-client-id"
        processor = SpanToDFWRecordProcessor(client_id=client_id)

        span = Span(name="parameter-test-span",
                    context=SpanContext(),
                    attributes={
                        "nat.event_type": IntermediateStepType.LLM_START, "extra_attribute": "test_value"
                    })

        await processor.process(span)

        # Verify all parameters are passed correctly
        mock_span_to_dfw_record.assert_called_once_with(span=span, to_type=processor.output_type, client_id=client_id)

    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.span_to_dfw_record')
    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.logger')
    async def test_logging_for_unsupported_event_types(self, mock_logger, mock_span_to_dfw_record):
        """Test that unsupported event types are logged appropriately."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        unsupported_event = IntermediateStepType.WORKFLOW_END
        span = Span(name="logging-test-span", context=SpanContext(), attributes={"nat.event_type": unsupported_event})

        result = await processor.process(span)

        assert result is None
        mock_logger.debug.assert_called_once_with("Unsupported event type: '%s'", unsupported_event)
        mock_span_to_dfw_record.assert_not_called()


class TestProcessorIntegration:
    """Integration tests combining both processors."""

    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.span_to_dfw_record')
    async def test_span_to_dict_pipeline(self, mock_span_to_dfw_record):
        """Test a complete pipeline from Span to Dict through both processors."""
        # Mock span_to_dfw_record to return a DFW record
        dfw_record = MockDFWRecord(record_id="pipeline-test", name="Pipeline Test", value=999)
        mock_span_to_dfw_record.return_value = dfw_record

        # Create processors
        span_processor = SpanToDFWRecordProcessor(client_id="pipeline-client")
        dict_processor = DFWToDictProcessor()

        # Create test span
        span = Span(name="pipeline-span",
                    context=SpanContext(),
                    attributes={"nat.event_type": IntermediateStepType.LLM_START})

        # Process through pipeline
        intermediate_result = await span_processor.process(span)
        final_result = await dict_processor.process(intermediate_result)

        # Verify results
        assert intermediate_result == dfw_record
        assert isinstance(final_result, dict)
        assert final_result["record_id"] == "pipeline-test"
        assert final_result["name"] == "Pipeline Test"
        assert final_result["value"] == 999

    async def test_span_processor_none_to_dict_processor(self):
        """Test pipeline when span processor returns None."""
        span_processor = SpanToDFWRecordProcessor(client_id="test")
        dict_processor = DFWToDictProcessor()

        # Create span that will return None (unsupported event type)
        span = Span(name="none-test-span",
                    context=SpanContext(),
                    attributes={"nat.event_type": IntermediateStepType.CUSTOM_END})

        # Process through pipeline
        intermediate_result = await span_processor.process(span)
        final_result = await dict_processor.process(intermediate_result)

        # Verify results
        assert intermediate_result is None
        assert final_result == {}


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    async def test_dfw_processor_with_invalid_json_structure(self):
        """Test DFWToDictProcessor handles edge cases in JSON serialization."""

        class ProblematicModel(BaseModel):
            record_id: str
            data: Any  # Could contain complex nested structures

            def model_dump_json(self, **kwargs):
                # Return valid JSON but with edge case structure
                return '{"record_id": "test", "data": null, "extra": {"nested": [1, 2, 3]}}'

        processor = DFWToDictProcessor()
        model = ProblematicModel(record_id="test", data={"complex": "data"})

        result = await processor.process(model)

        expected = {"record_id": "test", "data": None, "extra": {"nested": [1, 2, 3]}}
        assert result == expected

    async def test_span_processor_with_different_intermediate_step_types(self):
        """Test SpanToDFWRecordProcessor with various IntermediateStepType values."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        # Test all unsupported types
        unsupported_types = [
            IntermediateStepType.LLM_END,
            IntermediateStepType.LLM_NEW_TOKEN,
            IntermediateStepType.TOOL_START,
            IntermediateStepType.TOOL_END,
            IntermediateStepType.WORKFLOW_START,
            IntermediateStepType.WORKFLOW_END,
            IntermediateStepType.TASK_START,
            IntermediateStepType.TASK_END,
            IntermediateStepType.FUNCTION_START,
            IntermediateStepType.FUNCTION_END,
            IntermediateStepType.CUSTOM_START,
            IntermediateStepType.CUSTOM_END,
            IntermediateStepType.SPAN_START,
            IntermediateStepType.SPAN_CHUNK,
            IntermediateStepType.SPAN_END,
        ]

        for event_type in unsupported_types:
            span = Span(name=f"test-{event_type.value}",
                        context=SpanContext(),
                        attributes={"nat.event_type": event_type})

            result = await processor.process(span)
            assert result is None, f"Expected None for event type {event_type}"

    def test_processor_type_introspection(self):
        """Test type introspection capabilities of both processors."""
        dfw_processor = DFWToDictProcessor()
        span_processor = SpanToDFWRecordProcessor(client_id="test")

        # DFWToDictProcessor should have dict as output_type
        assert dfw_processor.output_type is dict

        # SpanToDFWRecordProcessor should have Span as input_type
        assert span_processor.input_type is Span

    @patch('nat.plugins.data_flywheel.observability.processor.dfw_record_processor.span_to_dfw_record')
    async def test_span_processor_cast_behavior(self, mock_span_to_dfw_record):
        """Test that the cast operation works correctly for type safety."""
        processor = SpanToDFWRecordProcessor(client_id="test")

        # Mock a return value that should be cast
        mock_record = MockTargetRecord(target_id="cast-test", converted_data="cast-data")
        mock_span_to_dfw_record.return_value = mock_record

        span = Span(name="cast-test-span",
                    context=SpanContext(),
                    attributes={"nat.event_type": IntermediateStepType.LLM_START})

        result = await processor.process(span)

        # Result should be properly cast and returned
        assert result is mock_record
        assert isinstance(result, MockTargetRecord)
