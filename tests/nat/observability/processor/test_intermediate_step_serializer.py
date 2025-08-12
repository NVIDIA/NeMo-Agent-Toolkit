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

import json
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
from nat.observability.processor.intermediate_step_serializer import IntermediateStepSerializer
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel


def create_test_intermediate_step(parent_id="root",
                                  function_name="test_function",
                                  function_id="test_id",
                                  **payload_kwargs):
    """Helper function to create IntermediateStep with proper structure for tests."""
    payload = IntermediateStepPayload(**payload_kwargs)
    function_ancestry = InvocationNode(function_name=function_name, function_id=function_id, parent_id=None)
    return IntermediateStep(parent_id=parent_id, function_ancestry=function_ancestry, payload=payload)


class TestIntermediateStepSerializerBasicFunctionality:
    """Test basic functionality of the IntermediateStepSerializer."""

    def test_serializer_is_processor_subclass(self):
        """Test that IntermediateStepSerializer is a proper subclass of Processor."""
        serializer = IntermediateStepSerializer()
        assert hasattr(serializer, 'process')
        assert hasattr(serializer, 'input_type')
        assert hasattr(serializer, 'output_type')
        assert serializer.input_type == IntermediateStep
        assert serializer.output_type == str

    def test_serializer_has_serialize_mixin(self):
        """Test that IntermediateStepSerializer has SerializeMixin functionality."""
        serializer = IntermediateStepSerializer()
        assert hasattr(serializer, '_serialize_payload')
        assert hasattr(serializer, '_process_streaming_output')

    @pytest.mark.asyncio
    async def test_basic_serialization(self):
        """Test basic serialization of an IntermediateStep."""
        # Create a simple IntermediateStep
        step = create_test_intermediate_step(event_type=IntermediateStepType.LLM_START,
                                             framework=LLMFrameworkEnum.LANGCHAIN,
                                             name="test_llm")

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        # Verify the result is a string
        assert isinstance(result, str)

        # Verify it's valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

        # Verify key fields are present
        assert 'payload' in parsed
        assert parsed['payload']['event_type'] == 'LLM_START'
        assert parsed['payload']['framework'] == 'langchain'
        assert parsed['payload']['name'] == 'test_llm'


class TestIntermediateStepSerializerWithDifferentData:
    """Test serialization with different types of intermediate step data."""

    @pytest.mark.asyncio
    async def test_serialization_with_stream_event_data(self):
        """Test serialization with StreamEventData."""
        stream_data = StreamEventData(input="test input", output="test output", chunk="test chunk")
        step = create_test_intermediate_step(event_type=IntermediateStepType.LLM_NEW_TOKEN, data=stream_data)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        parsed = json.loads(result)
        assert 'data' in parsed['payload']
        assert parsed['payload']['data']['input'] == 'test input'
        assert parsed['payload']['data']['output'] == 'test output'
        assert parsed['payload']['data']['chunk'] == 'test chunk'

    @pytest.mark.asyncio
    async def test_serialization_with_trace_metadata(self):
        """Test serialization with TraceMetadata."""
        metadata = TraceMetadata(chat_responses=["response1", "response2"],
                                 chat_inputs=["input1", "input2"],
                                 provided_metadata={"key": "value"})
        step = create_test_intermediate_step(event_type=IntermediateStepType.TOOL_START, metadata=metadata)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        parsed = json.loads(result)
        assert 'metadata' in parsed['payload']
        assert parsed['payload']['metadata']['chat_responses'] == ["response1", "response2"]
        assert parsed['payload']['metadata']['provided_metadata'] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_serialization_with_usage_info(self):
        """Test serialization with UsageInfo."""
        token_usage = TokenUsageBaseModel(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage_info = UsageInfo(token_usage=token_usage, num_llm_calls=1, seconds_between_calls=2)
        step = create_test_intermediate_step(event_type=IntermediateStepType.LLM_END, usage_info=usage_info)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        parsed = json.loads(result)
        assert 'usage_info' in parsed['payload']
        assert parsed['payload']['usage_info']['token_usage']['prompt_tokens'] == 100
        assert parsed['payload']['usage_info']['num_llm_calls'] == 1

    @pytest.mark.asyncio
    async def test_serialization_with_invocation_node(self):
        """Test serialization with function ancestry (InvocationNode)."""
        invocation_node = InvocationNode(function_name="test_function",
                                         function_id="test_id_123",
                                         parent_id="parent_id_456")
        payload = IntermediateStepPayload(event_type=IntermediateStepType.FUNCTION_START)
        step = IntermediateStep(parent_id="root", function_ancestry=invocation_node, payload=payload)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        parsed = json.loads(result)
        assert 'function_ancestry' in parsed
        assert parsed['function_ancestry']['function_name'] == 'test_function'
        assert parsed['function_ancestry']['function_id'] == 'test_id_123'

    @pytest.mark.asyncio
    async def test_serialization_with_complex_nested_data(self):
        """Test serialization with complex nested data structures."""
        complex_data = StreamEventData(input={"nested": {
            "key": "value", "list": [1, 2, 3]
        }},
                                       output={"result": ["item1", "item2"]},
                                       chunk={"partial": "data"})
        metadata = TraceMetadata(chat_responses=[{
            "role": "assistant", "content": "Hello"
        }],
                                 provided_metadata={
                                     "nested_dict": {
                                         "a": 1, "b": {
                                             "c": 2
                                         }
                                     }, "list_of_dicts": [{
                                         "x": 1
                                     }, {
                                         "y": 2
                                     }]
                                 })
        step = create_test_intermediate_step(event_type=IntermediateStepType.WORKFLOW_START,
                                             name="complex_workflow",
                                             tags=["tag1", "tag2"],
                                             data=complex_data,
                                             metadata=metadata)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        # Verify it's valid JSON with complex structure
        parsed = json.loads(result)
        assert parsed['payload']['data']['input']['nested']['key'] == 'value'
        assert parsed['payload']['metadata']['provided_metadata']['nested_dict']['b']['c'] == 2


class TestIntermediateStepSerializerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_serialization_with_minimal_data(self):
        """Test serialization with minimal required data."""
        step = create_test_intermediate_step(event_type=IntermediateStepType.CUSTOM_START)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        parsed = json.loads(result)
        assert 'payload' in parsed
        assert parsed['payload']['event_type'] == 'CUSTOM_START'
        # Should have default values
        assert 'event_timestamp' in parsed['payload']
        assert 'UUID' in parsed['payload']

    @pytest.mark.asyncio
    async def test_serialization_with_none_values(self):
        """Test serialization handles None values correctly."""
        payload = IntermediateStepPayload(event_type=IntermediateStepType.TASK_END,
                                          framework=None,
                                          name=None,
                                          tags=None,
                                          metadata=None,
                                          data=None,
                                          usage_info=None)
        # function_ancestry cannot be None, so provide a minimal InvocationNode
        function_ancestry = InvocationNode(function_name="test_function", function_id="test_id", parent_id=None)
        step = IntermediateStep(parent_id="root", function_ancestry=function_ancestry, payload=payload)

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        parsed = json.loads(result)
        assert parsed['function_ancestry']['function_name'] == 'test_function'
        assert parsed['function_ancestry']['function_id'] == 'test_id'
        assert parsed['payload']['framework'] is None
        assert parsed['payload']['name'] is None


class TestIntermediateStepSerializerErrorHandling:
    """Test error handling in serialization."""

    @pytest.mark.asyncio
    async def test_serialization_with_mock_error_handling(self):
        """Test that serialization falls back to string representation on errors."""
        step = create_test_intermediate_step(event_type=IntermediateStepType.LLM_START)

        serializer = IntermediateStepSerializer()

        # Mock _serialize_payload to return a string fallback (testing the SerializeMixin behavior)
        with patch.object(serializer, '_serialize_payload') as mock_serialize:
            # The SerializeMixin should catch exceptions and return string representation
            mock_serialize.return_value = (str(step), False)

            result = await serializer.process(step)
            assert isinstance(result, str)
            mock_serialize.assert_called_once_with(step)

    @pytest.mark.asyncio
    async def test_process_method_signature(self):
        """Test that the process method has the correct signature and behavior."""
        serializer = IntermediateStepSerializer()

        # Verify the method exists and is async
        assert hasattr(serializer, 'process')
        import inspect
        assert inspect.iscoroutinefunction(serializer.process)

    def test_mixin_integration(self):
        """Test that the SerializeMixin integration works correctly."""
        serializer = IntermediateStepSerializer()

        # Test _serialize_payload directly with a simple object
        simple_dict = {"key": "value"}
        result, is_json = serializer._serialize_payload(simple_dict)

        assert isinstance(result, str)
        assert is_json is True
        assert json.loads(result) == simple_dict


class TestIntermediateStepSerializerRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_llm_conversation_flow_serialization(self):
        """Test serialization of a typical LLM conversation flow."""
        # Create a sequence of steps like a real conversation
        steps = []

        # LLM Start
        steps.append(
            create_test_intermediate_step(event_type=IntermediateStepType.LLM_START,
                                          framework=LLMFrameworkEnum.LANGCHAIN,
                                          name="gpt-4",
                                          data=StreamEventData(input="What is the weather today?")))

        # LLM Tokens
        for i in range(3):
            steps.append(
                create_test_intermediate_step(event_type=IntermediateStepType.LLM_NEW_TOKEN,
                                              framework=LLMFrameworkEnum.LANGCHAIN,
                                              name="gpt-4",
                                              data=StreamEventData(chunk=f"Token_{i}")))

        # LLM End
        steps.append(
            create_test_intermediate_step(event_type=IntermediateStepType.LLM_END,
                                          framework=LLMFrameworkEnum.LANGCHAIN,
                                          name="gpt-4",
                                          data=StreamEventData(input="What is the weather today?",
                                                               output="I'll need to check the weather for you."),
                                          usage_info=UsageInfo(token_usage=TokenUsageBaseModel(prompt_tokens=20,
                                                                                               completion_tokens=15,
                                                                                               total_tokens=35),
                                                               num_llm_calls=1)))

        serializer = IntermediateStepSerializer()

        # Serialize each step
        serialized_steps = []
        for step in steps:
            result = await serializer.process(step)
            serialized_steps.append(json.loads(result))

        # Verify the sequence
        assert len(serialized_steps) == 5
        assert serialized_steps[0]['payload']['event_type'] == 'LLM_START'
        assert serialized_steps[1]['payload']['event_type'] == 'LLM_NEW_TOKEN'
        assert serialized_steps[4]['payload']['event_type'] == 'LLM_END'
        assert serialized_steps[4]['payload']['usage_info']['token_usage']['total_tokens'] == 35

    @pytest.mark.asyncio
    async def test_tool_execution_serialization(self):
        """Test serialization of tool execution steps."""
        # Tool Start
        tool_start = create_test_intermediate_step(event_type=IntermediateStepType.TOOL_START,
                                                   name="weather_tool",
                                                   data=StreamEventData(input={
                                                       "location": "New York", "units": "fahrenheit"
                                                   }))

        # Tool End
        tool_end = create_test_intermediate_step(event_type=IntermediateStepType.TOOL_END,
                                                 name="weather_tool",
                                                 data=StreamEventData(input={
                                                     "location": "New York", "units": "fahrenheit"
                                                 },
                                                                      output={
                                                                          "temperature": 72, "condition": "sunny"
                                                                      }))

        serializer = IntermediateStepSerializer()

        start_result = await serializer.process(tool_start)
        end_result = await serializer.process(tool_end)

        start_parsed = json.loads(start_result)
        end_parsed = json.loads(end_result)

        assert start_parsed['payload']['event_type'] == 'TOOL_START'
        assert start_parsed['payload']['data']['input']['location'] == 'New York'
        assert end_parsed['payload']['data']['output']['temperature'] == 72

    @pytest.mark.asyncio
    async def test_workflow_hierarchy_serialization(self):
        """Test serialization of workflow with function hierarchy."""
        child_node = InvocationNode(function_name="sub_task", function_id="sub_456", parent_id="main_123")

        workflow_step = IntermediateStep(
            parent_id="root",
            function_ancestry=child_node,
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.WORKFLOW_START,
                name="complex_workflow",
                metadata=TraceMetadata(provided_metadata={
                    "workflow_config": {
                        "max_iterations": 5
                    }, "context": {
                        "user_id": "12345"
                    }
                })))

        serializer = IntermediateStepSerializer()
        result = await serializer.process(workflow_step)

        parsed = json.loads(result)
        assert parsed['function_ancestry']['function_name'] == 'sub_task'
        assert parsed['function_ancestry']['parent_id'] == 'main_123'
        assert parsed['payload']['metadata']['provided_metadata']['workflow_config']['max_iterations'] == 5


class TestIntermediateStepSerializerTypeIntrospection:
    """Test type introspection capabilities inherited from Processor."""

    def test_type_introspection(self):
        """Test that type introspection works correctly."""
        serializer = IntermediateStepSerializer()

        assert serializer.input_type == IntermediateStep
        assert serializer.output_type == str
        assert serializer.input_class == IntermediateStep
        assert serializer.output_class == str

    def test_processor_inheritance_properties(self):
        """Test that all processor properties are available."""
        serializer = IntermediateStepSerializer()

        # Should have Processor properties
        assert hasattr(serializer, 'input_type')
        assert hasattr(serializer, 'output_type')
        assert hasattr(serializer, 'input_class')
        assert hasattr(serializer, 'output_class')

        # Should have SerializeMixin methods
        assert hasattr(serializer, '_serialize_payload')
        assert hasattr(serializer, '_process_streaming_output')

        # Should have the main process method
        assert hasattr(serializer, 'process')


class TestIntermediateStepSerializerPerformance:
    """Test performance characteristics of serialization."""

    @pytest.mark.asyncio
    async def test_serialization_of_large_data(self):
        """Test serialization performance with large data structures."""
        # Create a large data structure
        large_input = {"data": list(range(1000))}
        large_output = {"results": [{"id": i, "value": f"item_{i}"} for i in range(100)]}

        step = create_test_intermediate_step(event_type=IntermediateStepType.FUNCTION_END,
                                             data=StreamEventData(input=large_input, output=large_output))

        serializer = IntermediateStepSerializer()
        result = await serializer.process(step)

        # Verify it serializes correctly even with large data
        parsed = json.loads(result)
        assert len(parsed['payload']['data']['input']['data']) == 1000
        assert len(parsed['payload']['data']['output']['results']) == 100
        assert parsed['payload']['data']['output']['results'][0]['id'] == 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_serializations(self):
        """Test multiple sequential serializations work correctly."""
        serializer = IntermediateStepSerializer()

        # Create multiple different steps
        steps = []
        for i in range(10):
            steps.append(
                create_test_intermediate_step(event_type=IntermediateStepType.CUSTOM_START,
                                              name=f"step_{i}",
                                              data=StreamEventData(input=f"input_{i}", output=f"output_{i}")))

        # Serialize all steps
        results = []
        for step in steps:
            result = await serializer.process(step)
            results.append(result)

        # Verify all serializations worked
        assert len(results) == 10
        for i, result in enumerate(results):
            parsed = json.loads(result)
            assert parsed['payload']['name'] == f'step_{i}'
            assert parsed['payload']['data']['input'] == f'input_{i}'
