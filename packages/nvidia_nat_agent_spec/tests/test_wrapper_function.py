# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for AgentSpecWrapperFunction methods."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from nat.plugins.agent_spec.agent_spec_workflow import (
    AgentSpecWrapperFunction,
    AgentSpecWrapperInput,
    AgentSpecWrapperOutput,
)


class TestAgentSpecWrapperFunctionConvertInput:
    """Test cases for _convert_input() method."""

    def test_convert_input_string(self, wrapper_function):
        """Test _convert_input() with string input - most common use case."""
        result = wrapper_function._convert_input("Hello, world!")
        assert isinstance(result, AgentSpecWrapperInput)
        assert isinstance(result.messages, list)
        assert len(result.messages) == 1
        assert result.messages[0].content == "Hello, world!"

    def test_convert_input_list(self, wrapper_function):
        """Test _convert_input() with list input - common use case."""
        result = wrapper_function._convert_input(["Hello", "World"])
        assert isinstance(result, AgentSpecWrapperInput)
        assert isinstance(result.messages, list)
        assert len(result.messages) == 2
        assert result.messages[0].content == "Hello"
        assert result.messages[1].content == "World"


class TestAgentSpecWrapperFunctionAInvoke:
    """Test cases for _ainvoke() method."""

    @pytest.fixture
    def mock_input(self):
        """Create a mock AgentSpecWrapperInput."""
        return AgentSpecWrapperInput(
            messages=[HumanMessage(content="Test input")]
        )

    @pytest.fixture
    def mock_output_dict(self):
        """Create a mock output dictionary from graph."""
        return {
            "messages": [AIMessage(content="Test output")]
        }

    async def test_ainvoke_regular_graph(self, wrapper_function, mock_input, mock_output_dict):
        """Test _ainvoke() with regular graph (not async context manager)."""
        # Ensure graph is not treated as context manager by removing those attributes
        if hasattr(wrapper_function._graph, '__aenter__'):
            delattr(wrapper_function._graph, '__aenter__')
        if hasattr(wrapper_function._graph, '__aexit__'):
            delattr(wrapper_function._graph, '__aexit__')

        # Create async function that returns the dict
        async def mock_ainvoke(input_dict):
            return mock_output_dict
        wrapper_function._graph.ainvoke = mock_ainvoke

        result = await wrapper_function._ainvoke(mock_input)

        assert isinstance(result, AgentSpecWrapperOutput)
        assert len(result.messages) == 1
        assert result.messages[0].content == "Test output"

    async def test_ainvoke_async_context_manager(self, wrapper_function, mock_input, mock_output_dict):
        """Test _ainvoke() with async context manager graph."""
        inner_graph = MagicMock()
        inner_graph.ainvoke = AsyncMock(return_value=mock_output_dict)

        wrapper_function._graph.__aenter__ = AsyncMock(return_value=inner_graph)
        wrapper_function._graph.__aexit__ = AsyncMock(return_value=None)

        result = await wrapper_function._ainvoke(mock_input)

        assert isinstance(result, AgentSpecWrapperOutput)
        assert result.messages[0].content == "Test output"
        wrapper_function._graph.__aenter__.assert_called_once()
        wrapper_function._graph.__aexit__.assert_called_once()
        inner_graph.ainvoke.assert_called_once()

    @pytest.mark.parametrize("use_context_manager", [False, True])
    async def test_ainvoke_error_handling(self, wrapper_function, mock_input, use_context_manager):
        """Test _ainvoke() error handling for both graph types."""
        original_error = ValueError("Graph execution failed")

        if use_context_manager:
            inner_graph = MagicMock()
            inner_graph.ainvoke = AsyncMock(side_effect=original_error)
            wrapper_function._graph.__aenter__ = AsyncMock(return_value=inner_graph)
            wrapper_function._graph.__aexit__ = AsyncMock(return_value=None)
        else:
            wrapper_function._graph.ainvoke = AsyncMock(side_effect=original_error)

        with pytest.raises(RuntimeError, match="Error executing Agent-Spec workflow"):
            await wrapper_function._ainvoke(mock_input)

        if use_context_manager:
            wrapper_function._graph.__aexit__.assert_called_once()


class TestAgentSpecWrapperFunctionAStream:
    """Test cases for _astream() method."""

    @pytest.fixture
    def mock_input(self):
        """Create a mock AgentSpecWrapperInput."""
        return AgentSpecWrapperInput(
            messages=[HumanMessage(content="Test input")]
        )

    async def test_astream_regular_graph(self, wrapper_function, mock_input):
        """Test _astream() with regular graph (not async context manager)."""
        # Ensure graph is not treated as context manager
        if hasattr(wrapper_function._graph, '__aenter__'):
            delattr(wrapper_function._graph, '__aenter__')
        if hasattr(wrapper_function._graph, '__aexit__'):
            delattr(wrapper_function._graph, '__aexit__')

        async def mock_stream():
            yield {"messages": [AIMessage(content="Output 1")]}
            yield {"messages": [AIMessage(content="Output 2")]}
            yield {"messages": [AIMessage(content="Output 3")]}

        # Create a function that returns the async generator
        def astream_func(input_dict):
            return mock_stream()
        wrapper_function._graph.astream = astream_func

        results = []
        async for output in wrapper_function._astream(mock_input):
            results.append(output)

        assert len(results) == 3
        assert all(isinstance(r, AgentSpecWrapperOutput) for r in results)
        assert [r.messages[0].content for r in results] == ["Output 1", "Output 2", "Output 3"]

    async def test_astream_async_context_manager(self, wrapper_function, mock_input):
        """Test _astream() with async context manager graph."""
        async def mock_stream():
            yield {"messages": [AIMessage(content="Streamed output")]}

        inner_graph = MagicMock()
        inner_graph.astream = lambda input_dict: mock_stream()

        wrapper_function._graph.__aenter__ = AsyncMock(return_value=inner_graph)
        wrapper_function._graph.__aexit__ = AsyncMock(return_value=None)

        results = []
        async for output in wrapper_function._astream(mock_input):
            results.append(output)

        assert len(results) == 1
        assert results[0].messages[0].content == "Streamed output"
        wrapper_function._graph.__aenter__.assert_called_once()
        wrapper_function._graph.__aexit__.assert_called_once()

    @pytest.mark.parametrize("use_context_manager", [False, True])
    async def test_astream_error_handling(self, wrapper_function, mock_input, use_context_manager):
        """Test _astream() error handling for both graph types."""
        original_error = ValueError("Streaming failed")

        async def failing_stream():
            yield {"messages": [AIMessage(content="First")]}
            raise original_error

        if use_context_manager:
            inner_graph = MagicMock()
            inner_graph.astream = AsyncMock(return_value=failing_stream())
            wrapper_function._graph.__aenter__ = AsyncMock(return_value=inner_graph)
            wrapper_function._graph.__aexit__ = AsyncMock(return_value=None)
        else:
            wrapper_function._graph.astream = AsyncMock(return_value=failing_stream())

        with pytest.raises(RuntimeError, match="Error streaming Agent-Spec workflow"):
            async for _ in wrapper_function._astream(mock_input):
                pass

        if use_context_manager:
            wrapper_function._graph.__aexit__.assert_called_once()


class TestAgentSpecWrapperFunctionConvertToStr:
    """Test cases for convert_to_str() static method."""

    def test_convert_to_str_with_messages(self):
        """Test convert_to_str() returns last message's text."""
        output = AgentSpecWrapperOutput(
            messages=[
                AIMessage(content="First message"),
                AIMessage(content="Second message"),
                AIMessage(content="Last message"),
            ]
        )
        result = AgentSpecWrapperFunction.convert_to_str(output)
        assert result == "Last message"

    def test_convert_to_str_empty_messages(self):
        """Test convert_to_str() with empty messages."""
        output = AgentSpecWrapperOutput(messages=[])
        result = AgentSpecWrapperFunction.convert_to_str(output)
        assert result == ""

    def test_convert_to_str_mixed_message_types(self):
        """Test convert_to_str() with mixed message types."""
        output = AgentSpecWrapperOutput(
            messages=[
                SystemMessage(content="System"),
                HumanMessage(content="Human"),
                AIMessage(content="AI"),
            ]
        )
        result = AgentSpecWrapperFunction.convert_to_str(output)
        assert result == "AI"
