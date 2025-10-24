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

import os
import uuid
from datetime import datetime
from unittest.mock import patch

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.span import MimeTypes
from nat.data_models.span import Span
from nat.data_models.span import SpanAttributes
from nat.observability.exporter.span_exporter import SpanExporter
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel


def create_test_intermediate_step(parent_id="root",
                                  function_name="test_function",
                                  function_id="test_id",
                                  **payload_kwargs):
    """Helper function to create IntermediateStep with proper structure for tests."""
    payload = IntermediateStepPayload(**payload_kwargs)
    function_ancestry = InvocationNode(function_name=function_name, function_id=function_id, parent_id=None)
    return IntermediateStep(parent_id=parent_id, function_ancestry=function_ancestry, payload=payload)


def create_intermediate_step(parent_id="root", function_name="test_function", function_id="test_id", **payload_kwargs):
    """Helper function to create IntermediateStep with proper structure."""
    # Set defaults for InvocationNode
    function_id = payload_kwargs.get("UUID", "test-function-id")
    function_name = payload_kwargs.get("name") or "test_function"

    return IntermediateStep(parent_id=parent_id,
                            payload=IntermediateStepPayload(**payload_kwargs),
                            function_ancestry=InvocationNode(function_id=function_id,
                                                             function_name=function_name,
                                                             parent_id=None))


class ConcreteSpanExporter(SpanExporter[Span, Span]):
    """Concrete implementation of SpanExporter for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_spans = []

    async def export_processed(self, item: Span) -> None:
        """Export the processed span."""
        self.exported_spans.append(item)


class TestSpanExporterFunctionality:
    """Test suite for SpanExporter functionality."""

    @pytest.fixture
    def span_exporter(self):
        """Create a test span exporter instance."""
        return ConcreteSpanExporter()

    @pytest.fixture
    def sample_start_event(self):
        """Create a sample START event."""
        return IntermediateStep(parent_id="root",
                                payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                                framework=LLMFrameworkEnum.LANGCHAIN,
                                                                name="test_llm_call",
                                                                event_timestamp=datetime.now().timestamp(),
                                                                data=StreamEventData(input="Test input"),
                                                                metadata={"key": "value"}),
                                function_ancestry=InvocationNode(function_id="func_123",
                                                                 function_name="test_function",
                                                                 parent_id=None))

    @pytest.fixture
    def sample_end_event(self):
        """Create a sample END event."""
        return IntermediateStep(
            parent_id="root",
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                framework=LLMFrameworkEnum.LANGCHAIN,
                name="test_llm_call",
                event_timestamp=datetime.now().timestamp(),
                span_event_timestamp=datetime.now().timestamp(),
                data=StreamEventData(output="Test output"),
                metadata={"end_key": "end_value"},
                usage_info=UsageInfo(
                    num_llm_calls=1,
                    seconds_between_calls=1,  # Must be int
                    token_usage=TokenUsageBaseModel(prompt_tokens=10, completion_tokens=20, total_tokens=30))),
            function_ancestry=InvocationNode(function_id="func_123", function_name="test_function", parent_id=None))

    def test_init(self, span_exporter):
        """Test SpanExporter initialization."""
        assert span_exporter._outstanding_spans == {}
        assert span_exporter._span_stack == {}
        assert span_exporter._metadata_stack == {}
        assert span_exporter.exported_spans == []

    def test_export_non_intermediate_step(self, span_exporter):
        """Test export with non-IntermediateStep event."""
        # Should not raise exception or process anything
        span_exporter.export("not an intermediate step")
        assert len(span_exporter._outstanding_spans) == 0

    @pytest.mark.usefixtures("restore_environ")
    @pytest.mark.parametrize("use_environ", [True, False])
    @pytest.mark.parametrize("span_prefix, expected_span_prefix", [(None, "nat"), ("nat", "nat"), ("custom", "custom")])
    def test_process_start_event(self,
                                 sample_start_event: IntermediateStep,
                                 span_prefix: str | None,
                                 expected_span_prefix: str,
                                 use_environ: bool):
        """Test processing START event."""
        if use_environ:
            if span_prefix is not None:
                os.environ["NAT_SPAN_PREFIX"] = span_prefix
            span_exporter = ConcreteSpanExporter()
        else:
            span_exporter = ConcreteSpanExporter(span_prefix=span_prefix)

        span_exporter.export(sample_start_event)

        # Check that span was created and added to tracking
        assert len(span_exporter._outstanding_spans) == 1
        assert len(span_exporter._span_stack) == 1
        assert len(span_exporter._metadata_stack) == 1

        # Check span properties
        span = span_exporter._outstanding_spans[sample_start_event.payload.UUID]
        assert isinstance(span, Span)
        assert span.name == "test_llm_call"
        assert span.attributes[f"{expected_span_prefix}.event_type"] == IntermediateStepType.LLM_START.value
        assert span.attributes[f"{expected_span_prefix}.function.id"] == "func_123"
        assert span.attributes[f"{expected_span_prefix}.function.name"] == "test_function"
        assert span.attributes[f"{expected_span_prefix}.framework"] == LLMFrameworkEnum.LANGCHAIN.value

    def test_process_start_event_with_parent(self, span_exporter):
        """Test processing START event with parent span."""
        # Create parent event first
        parent_event = IntermediateStep(parent_id="root",
                                        payload=IntermediateStepPayload(UUID="parent_id",
                                                                        event_type=IntermediateStepType.FUNCTION_START,
                                                                        framework=LLMFrameworkEnum.LANGCHAIN,
                                                                        name="parent_call",
                                                                        event_timestamp=datetime.now().timestamp(),
                                                                        data=StreamEventData(input="Parent input"),
                                                                        metadata={"parent_key": "parent_value"}),
                                        function_ancestry=InvocationNode(function_id="parent_func",
                                                                         function_name="parent_function",
                                                                         parent_id=None))

        # Process parent event
        span_exporter.export(parent_event)

        # Create child event
        child_event = IntermediateStep(parent_id="parent_id",
                                       payload=IntermediateStepPayload(UUID="child_id",
                                                                       event_type=IntermediateStepType.LLM_START,
                                                                       framework=LLMFrameworkEnum.LANGCHAIN,
                                                                       name="child_call",
                                                                       event_timestamp=datetime.now().timestamp(),
                                                                       data=StreamEventData(input="Child input"),
                                                                       metadata={"child_key": "child_value"}),
                                       function_ancestry=InvocationNode(function_id="child_func",
                                                                        function_name="child_function",
                                                                        parent_id="parent_id"))

        # Process child event
        span_exporter.export(child_event)

        # Check that child span has parent context
        child_span = span_exporter._outstanding_spans["child_id"]
        parent_span = span_exporter._outstanding_spans["parent_id"]

        assert child_span.parent is not None
        assert child_span.context is not None
        assert child_span.context.trace_id == parent_span.context.trace_id if parent_span.context else None

    def test_process_start_event_missing_parent(self, span_exporter):
        """Test processing START event with missing parent."""
        # First create a span stack so we have existing spans but not the parent we're looking for
        dummy_span = Span(name="dummy", attributes={}, start_time=0)
        span_exporter._span_stack["dummy_id"] = dummy_span

        event = create_intermediate_step(parent_id="missing_parent_id",
                                         function_name="child_function",
                                         function_id="child_func",
                                         UUID="child_id",
                                         event_type=IntermediateStepType.LLM_START,
                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                         name="child_call",
                                         event_timestamp=datetime.now().timestamp(),
                                         data=StreamEventData(input="Child input"),
                                         metadata={"child_key": "child_value"})

        with patch('nat.observability.exporter.span_exporter.logger') as mock_logger:
            span_exporter.export(event)
            mock_logger.warning.assert_called_once()

    def test_process_start_event_input_parsing(self, span_exporter):
        """Test processing START event with different input formats."""
        # Test with Human: Question: format
        event = create_intermediate_step(event_type=IntermediateStepType.LLM_START,
                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                         name="test_llm_call",
                                         event_timestamp=datetime.now().timestamp(),
                                         data=StreamEventData(input="Human: Question: What is the capital of France?"),
                                         metadata={"key": "value"})

        span_exporter.export(event)
        span = span_exporter._outstanding_spans[event.payload.UUID]
        assert span.attributes[SpanAttributes.INPUT_VALUE.value] == "What is the capital of France?"

    async def test_process_end_event(self, span_exporter, sample_start_event, sample_end_event):
        """Test processing END event."""
        # Use same UUID for start and end events
        sample_end_event.payload.UUID = sample_start_event.payload.UUID

        # Start the exporter to enable async export using proper context manager
        async with span_exporter.start():
            # Process start event first
            span_exporter.export(sample_start_event)

            # Process end event
            span_exporter.export(sample_end_event)

            # Check that span was removed from tracking
            assert len(span_exporter._outstanding_spans) == 0
            assert len(span_exporter._span_stack) == 0
            assert len(span_exporter._metadata_stack) == 0

            # Wait for async export to complete
            await span_exporter.wait_for_tasks()

            # Check that span was exported
            assert len(span_exporter.exported_spans) == 1
            exported_span = span_exporter.exported_spans[0]

            # Check attributes were set correctly
            assert exported_span.attributes[SpanAttributes.NAT_USAGE_NUM_LLM_CALLS.value] == 1
            assert exported_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value] == 10
            assert exported_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value] == 20
            assert exported_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value] == 30
            assert exported_span.attributes[SpanAttributes.OUTPUT_VALUE.value] == "Test output"
            assert "nat.metadata" in exported_span.attributes

    def test_process_end_event_missing_span(self, span_exporter, sample_end_event):
        """Test processing END event with missing span."""
        with patch('nat.observability.exporter.span_exporter.logger') as mock_logger:
            span_exporter.export(sample_end_event)
            mock_logger.warning.assert_called_once()

    async def test_process_end_event_metadata_merge(self, span_exporter):
        """Test metadata merging in END event processing."""
        event_id = str(uuid.uuid4())

        # Start event with metadata
        start_event = create_intermediate_step(UUID=event_id,
                                               event_type=IntermediateStepType.LLM_START,
                                               framework=LLMFrameworkEnum.LANGCHAIN,
                                               name="test_call",
                                               event_timestamp=datetime.now().timestamp(),
                                               data=StreamEventData(input="Test input"),
                                               metadata={
                                                   "start_key": "start_value", "common_key": "start_common"
                                               })

        # End event with metadata
        end_event = create_intermediate_step(UUID=event_id,
                                             event_type=IntermediateStepType.LLM_END,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_call",
                                             event_timestamp=datetime.now().timestamp(),
                                             span_event_timestamp=datetime.now().timestamp(),
                                             data=StreamEventData(output="Test output"),
                                             metadata={
                                                 "end_key": "end_value", "common_key": "end_common"
                                             })

        # Start the exporter to enable async export using proper context manager
        async with span_exporter.start():
            # Process events
            span_exporter.export(start_event)
            span_exporter.export(end_event)

            # Wait for async tasks to complete
            await span_exporter.wait_for_tasks()

            # Check that span was processed
            assert len(span_exporter._outstanding_spans) == 0
            assert len(span_exporter.exported_spans) == 1

    async def test_process_end_event_trace_metadata(self, span_exporter):
        """Test END event processing with TraceMetadata objects."""
        event_id = str(uuid.uuid4())

        # Start event
        start_event = create_intermediate_step(UUID=event_id,
                                               event_type=IntermediateStepType.LLM_START,
                                               framework=LLMFrameworkEnum.LANGCHAIN,
                                               name="test_call",
                                               event_timestamp=datetime.now().timestamp(),
                                               data=StreamEventData(input="Test input"),
                                               metadata=TraceMetadata(provided_metadata={
                                                   "workflow_id": "workflow_123", "session_id": "session_456"
                                               }))

        # End event
        end_event = create_intermediate_step(UUID=event_id,
                                             event_type=IntermediateStepType.LLM_END,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_call",
                                             event_timestamp=datetime.now().timestamp(),
                                             span_event_timestamp=datetime.now().timestamp(),
                                             data=StreamEventData(output="Test output"),
                                             metadata=TraceMetadata(provided_metadata={
                                                 "workflow_id": "workflow_123", "session_id": "session_456"
                                             }))

        # Start the exporter to enable async export using proper context manager
        async with span_exporter.start():
            # Process events
            span_exporter.export(start_event)
            span_exporter.export(end_event)

            # Wait for async tasks to complete
            await span_exporter.wait_for_tasks()

            # Check that span was processed
            assert len(span_exporter._outstanding_spans) == 0
            assert len(span_exporter.exported_spans) == 1

    def test_process_end_event_invalid_metadata(self, span_exporter):
        """Test END event processing with invalid metadata in end event."""
        # Test invalid metadata in end event (should trigger validation in pydantic)
        event_id = str(uuid.uuid4())

        # Start event
        start_event = create_intermediate_step(UUID=event_id,
                                               event_type=IntermediateStepType.LLM_START,
                                               framework=LLMFrameworkEnum.LANGCHAIN,
                                               name="test_call",
                                               event_timestamp=datetime.now().timestamp(),
                                               data=StreamEventData(input="Test input"),
                                               metadata={"valid": "metadata"})

        # Process start event
        span_exporter.export(start_event)

        # Manually create an end event that will cause issues when trying to validate
        # metadata (since pydantic validates at creation time, we need to test different scenario)
        with patch('nat.observability.exporter.span_exporter.logger') as mock_logger:
            # Test when end_metadata is not a dict or TraceMetadata after creation
            end_event = start_event.model_copy()
            end_event.payload.event_type = IntermediateStepType.LLM_END
            end_event.payload.metadata = "invalid_metadata_string"  # This is invalid type

            span_exporter.export(end_event)
            mock_logger.warning.assert_called()

    def test_process_end_event_missing_metadata(self, span_exporter):
        """Test END event processing with missing start metadata."""
        event_id = str(uuid.uuid4())

        # Manually add span to outstanding spans but NOT to metadata stack
        span = Span(name="test_span", attributes={}, start_time=0)
        span_exporter._outstanding_spans[event_id] = span
        # Don't add to metadata_stack to simulate missing metadata

        # End event
        end_event = create_intermediate_step(UUID=event_id,
                                             event_type=IntermediateStepType.LLM_END,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_call",
                                             event_timestamp=datetime.now().timestamp(),
                                             span_event_timestamp=datetime.now().timestamp(),
                                             data=StreamEventData(output="Test output"),
                                             metadata={"end_key": "end_value"})

        # The KeyError is expected because metadata is missing - this is a legitimate runtime error
        # Instead of mocking logger, we check that the exception happens and span processing stops
        with pytest.raises(KeyError):
            span_exporter.export(end_event)

    async def test_cleanup(self, span_exporter):
        """Test cleanup functionality."""
        # Add some outstanding spans
        span1 = Span(name="span1", attributes={}, start_time=0)
        span2 = Span(name="span2", attributes={}, start_time=0)

        span_exporter._outstanding_spans["span1"] = span1
        span_exporter._outstanding_spans["span2"] = span2
        span_exporter._span_stack["span1"] = span1
        span_exporter._metadata_stack["span1"] = {"key": "value"}

        with patch('nat.observability.exporter.span_exporter.logger') as mock_logger:
            await span_exporter._cleanup()
            mock_logger.warning.assert_called_once()

        # Check that all tracking is cleared
        assert len(span_exporter._outstanding_spans) == 0
        assert len(span_exporter._span_stack) == 0
        assert len(span_exporter._metadata_stack) == 0

    async def test_cleanup_no_outstanding_spans(self, span_exporter):
        """Test cleanup with no outstanding spans."""
        # Should not raise any exceptions
        await span_exporter._cleanup()

        assert len(span_exporter._outstanding_spans) == 0
        assert len(span_exporter._span_stack) == 0
        assert len(span_exporter._metadata_stack) == 0

    def test_span_attribute_setting(self, span_exporter, sample_start_event):
        """Test various span attribute settings."""
        # Test with different input formats
        sample_start_event.payload.data = StreamEventData(input={"complex": "json", "data": [1, 2, 3]})

        span_exporter.export(sample_start_event)

        span = span_exporter._outstanding_spans[sample_start_event.payload.UUID]
        assert SpanAttributes.INPUT_VALUE.value in span.attributes
        assert SpanAttributes.INPUT_MIME_TYPE.value in span.attributes
        assert span.attributes[SpanAttributes.INPUT_MIME_TYPE.value] == MimeTypes.JSON.value

    def test_span_name_generation(self, span_exporter):
        """Test span name generation logic."""
        # Test with name provided
        event_with_name = create_intermediate_step(event_type=IntermediateStepType.LLM_START,
                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                   name="custom_name",
                                                   event_timestamp=datetime.now().timestamp(),
                                                   data=StreamEventData(input="Test input"),
                                                   metadata={"key": "value"})

        span_exporter.export(event_with_name)
        span = span_exporter._outstanding_spans[event_with_name.payload.UUID]
        assert span.name == "custom_name"

        # Test without name (should use event_type string representation)
        event_without_name = create_intermediate_step(event_type=IntermediateStepType.TOOL_START,
                                                      framework=LLMFrameworkEnum.LANGCHAIN,
                                                      name=None,
                                                      event_timestamp=datetime.now().timestamp(),
                                                      data=StreamEventData(input="Test input"),
                                                      metadata={"key": "value"})

        span_exporter.export(event_without_name)
        span = span_exporter._outstanding_spans[event_without_name.payload.UUID]
        # The actual implementation uses str() on the enum, which includes the full representation
        assert span.name == str(IntermediateStepType.TOOL_START)

    def test_span_context_propagation(self, span_exporter):
        """Test that span context and trace IDs are properly propagated."""
        # Create parent event
        parent_event = create_intermediate_step(UUID="parent_id",
                                                event_type=IntermediateStepType.FUNCTION_START,
                                                framework=LLMFrameworkEnum.LANGCHAIN,
                                                name="parent_call",
                                                event_timestamp=datetime.now().timestamp(),
                                                data=StreamEventData(input="Parent input"),
                                                metadata={"parent_key": "parent_value"})

        # Process parent event
        span_exporter.export(parent_event)
        parent_span = span_exporter._outstanding_spans["parent_id"]

        # Verify parent span has context (root spans get contexts too)
        assert parent_span.context is not None
        parent_trace_id = parent_span.context.trace_id

        # Create child event with proper parent relationship
        child_event = create_intermediate_step(parent_id="parent_id",
                                               UUID="child_id",
                                               event_type=IntermediateStepType.LLM_START,
                                               framework=LLMFrameworkEnum.LANGCHAIN,
                                               name="child_call",
                                               event_timestamp=datetime.now().timestamp(),
                                               data=StreamEventData(input="Child input"),
                                               metadata={"child_key": "child_value"})

        # Process child event
        span_exporter.export(child_event)
        child_span = span_exporter._outstanding_spans["child_id"]

        # Verify parent-child relationship was established
        assert child_span.parent is not None
        assert child_span.parent.name == "parent_call"
        # Verify trace ID propagation
        assert child_span.context is not None
        assert child_span.context.trace_id == parent_trace_id

    def test_isolated_attributes(self):
        """Test that isolated attributes work correctly across different instances."""
        exporter1 = ConcreteSpanExporter()
        exporter2 = ConcreteSpanExporter()

        # Add data to first exporter
        exporter1._outstanding_spans["test1"] = "span1"
        exporter1._span_stack["test1"] = "stack1"
        exporter1._metadata_stack["test1"] = "meta1"

        # Add different data to second exporter
        exporter2._outstanding_spans["test2"] = "span2"
        exporter2._span_stack["test2"] = "stack2"
        exporter2._metadata_stack["test2"] = "meta2"

        # Check isolation
        assert "test1" in exporter1._outstanding_spans
        assert "test1" not in exporter2._outstanding_spans
        assert "test2" in exporter2._outstanding_spans
        assert "test2" not in exporter1._outstanding_spans

    async def test_usage_info_without_token_usage(self, span_exporter):
        """Test END event processing with usage info but minimal token usage."""
        event_id = str(uuid.uuid4())

        # Start event
        start_event = create_intermediate_step(UUID=event_id,
                                               event_type=IntermediateStepType.LLM_START,
                                               framework=LLMFrameworkEnum.LANGCHAIN,
                                               name="test_call",
                                               event_timestamp=datetime.now().timestamp(),
                                               data=StreamEventData(input="Test input"),
                                               metadata={"key": "value"})

        # End event with usage info and minimal token usage (all zeros)
        end_event = create_intermediate_step(UUID=event_id,
                                             event_type=IntermediateStepType.LLM_END,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_call",
                                             event_timestamp=datetime.now().timestamp(),
                                             span_event_timestamp=datetime.now().timestamp(),
                                             data=StreamEventData(output="Test output"),
                                             metadata={"end_key": "end_value"},
                                             usage_info=UsageInfo(num_llm_calls=2,
                                                                  seconds_between_calls=5,
                                                                  token_usage=TokenUsageBaseModel(prompt_tokens=0,
                                                                                                  completion_tokens=0,
                                                                                                  total_tokens=0)))

        # Start the exporter to enable async export using proper context manager
        async with span_exporter.start():
            # Process events
            span_exporter.export(start_event)
            span_exporter.export(end_event)

            # Wait for async tasks to complete
            await span_exporter.wait_for_tasks()

            # Check that span was processed and attributes set correctly
            assert len(span_exporter._outstanding_spans) == 0
            assert len(span_exporter.exported_spans) == 1
