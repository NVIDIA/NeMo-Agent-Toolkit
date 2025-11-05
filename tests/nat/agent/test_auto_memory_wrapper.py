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

import datetime
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from nat.agent.auto_memory_wrapper.agent import AutoMemoryWrapperGraph
from nat.agent.auto_memory_wrapper.state import AutoMemoryWrapperState
from nat.builder.context import Context
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChoice
from nat.data_models.api_server import ChoiceMessage
from nat.data_models.api_server import Usage
from nat.data_models.api_server import UserMessageContentRoleType
from nat.memory.models import MemoryItem


@pytest.fixture
def mock_memory_editor():
    """Create a mock MemoryEditor for testing."""
    editor = AsyncMock()
    editor.add_items = AsyncMock()
    editor.search = AsyncMock(return_value=[])
    return editor


@pytest.fixture
def mock_inner_agent():
    """Create a mock inner agent function for testing."""
    mock_fn = Mock()

    async def _ainvoke(chat_request: ChatRequest):
        # Simulate agent processing and return a ChatResponse
        return ChatResponse(id="test-response-id",
                            created=datetime.datetime.now(),
                            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                            choices=[
                                ChatResponseChoice(index=0,
                                                   message=ChoiceMessage(role=UserMessageContentRoleType.ASSISTANT,
                                                                         content="Agent response"))
                            ])

    # Wrap the async function in AsyncMock so we can track calls
    mock_fn.ainvoke = AsyncMock(side_effect=_ainvoke)
    return mock_fn


@pytest.fixture
def mock_context():
    """Create a mock Context for testing."""
    context = Mock(spec=Context)
    context.user_manager = None
    context.metadata = None
    return context


@pytest.fixture
def wrapper_graph(mock_inner_agent, mock_memory_editor, mock_context):
    """Create an AutoMemoryWrapperGraph instance for testing."""
    with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
        return AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                      memory_editor=mock_memory_editor,
                                      save_user_messages=True,
                                      retrieve_memory=True,
                                      save_ai_responses=True)


class TestAutoMemoryWrapperState:
    """Test AutoMemoryWrapperState schema and initialization."""

    def test_state_initialization_empty(self):
        """Test that AutoMemoryWrapperState initializes with empty messages."""
        state = AutoMemoryWrapperState()
        assert isinstance(state.messages, list)
        assert len(state.messages) == 0

    def test_state_initialization_with_messages(self):
        """Test AutoMemoryWrapperState initialization with provided messages."""
        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
        state = AutoMemoryWrapperState(messages=messages)
        assert state.messages == messages
        assert len(state.messages) == 2


class TestAutoMemoryWrapperGraph:
    """Test AutoMemoryWrapperGraph initialization and core functionality."""

    def test_initialization(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test AutoMemoryWrapperGraph initialization with all features enabled."""
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=True,
                                             retrieve_memory=True,
                                             save_ai_responses=True,
                                             search_params={"top_k": 5},
                                             add_params={"ignore_roles": ["assistant"]})

        assert wrapper.inner_agent_fn == mock_inner_agent
        assert wrapper.memory_editor == mock_memory_editor
        assert wrapper.save_user_messages is True
        assert wrapper.retrieve_memory is True
        assert wrapper.save_ai_responses is True
        assert wrapper.search_params == {"top_k": 5}
        assert wrapper.add_params == {"ignore_roles": ["assistant"]}

    def test_get_wrapper_node_count_all_enabled(self, wrapper_graph):
        """Test wrapper node count with all features enabled."""
        count = wrapper_graph.get_wrapper_node_count()
        assert count == 4  # capture_user + retrieve + inner + capture_ai

    def test_get_wrapper_node_count_minimal(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test wrapper node count with minimal features."""
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=False,
                                             retrieve_memory=False,
                                             save_ai_responses=False)
        count = wrapper.get_wrapper_node_count()
        assert count == 1  # only inner_agent

    def test_get_user_id_default(self, wrapper_graph):
        """Test user ID extraction defaults to 'default_user'."""
        user_id = wrapper_graph._get_user_id_from_context()
        assert user_id == "default_user"

    def test_get_user_id_from_header(self, wrapper_graph, mock_context):
        """Test user ID extraction from X-User-ID header."""
        mock_context.metadata = Mock()
        mock_context.metadata.headers = {"x-user-id": "test-user-123"}
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper_graph._context = mock_context
            user_id = wrapper_graph._get_user_id_from_context()
        assert user_id == "test-user-123"

    def test_get_user_id_from_user_manager(self, wrapper_graph, mock_context):
        """Test user ID extraction from user_manager."""
        mock_user_manager = Mock()
        mock_user_manager.get_id.return_value = "user-from-manager"
        mock_context.user_manager = mock_user_manager
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper_graph._context = mock_context
            user_id = wrapper_graph._get_user_id_from_context()
        assert user_id == "user-from-manager"

    def test_langchain_message_to_nat_message_human(self):
        """Test conversion of HumanMessage to NAT Message."""
        lc_message = HumanMessage(content="Hello")
        nat_message = AutoMemoryWrapperGraph._langchain_message_to_nat_message(lc_message)
        assert nat_message.role == UserMessageContentRoleType.USER
        assert nat_message.content == "Hello"

    def test_langchain_message_to_nat_message_ai(self):
        """Test conversion of AIMessage to NAT Message."""
        lc_message = AIMessage(content="Hi there")
        nat_message = AutoMemoryWrapperGraph._langchain_message_to_nat_message(lc_message)
        assert nat_message.role == UserMessageContentRoleType.ASSISTANT
        assert nat_message.content == "Hi there"

    def test_langchain_message_to_nat_message_system(self):
        """Test conversion of SystemMessage to NAT Message."""
        lc_message = SystemMessage(content="System prompt")
        nat_message = AutoMemoryWrapperGraph._langchain_message_to_nat_message(lc_message)
        assert nat_message.role == UserMessageContentRoleType.SYSTEM
        assert nat_message.content == "System prompt"

    async def test_capture_user_message_node(self, wrapper_graph, mock_memory_editor):
        """Test capture_user_message_node saves user messages."""
        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Test message")])
        result = await wrapper_graph.capture_user_message_node(state)

        assert result == state
        mock_memory_editor.add_items.assert_called_once()
        call_args = mock_memory_editor.add_items.call_args
        items = call_args[0][0]
        assert len(items) == 1
        assert items[0].conversation == [{"role": "user", "content": "Test message"}]
        assert items[0].user_id == "default_user"

    async def test_capture_user_message_node_disabled(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test capture_user_message_node when disabled."""
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=False,
                                             retrieve_memory=False,
                                             save_ai_responses=False)

        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Test")])
        result = await wrapper.capture_user_message_node(state)

        assert result == state
        mock_memory_editor.add_items.assert_not_called()

    async def test_capture_user_message_node_empty_messages(self, wrapper_graph, mock_memory_editor):
        """Test capture_user_message_node with empty messages."""
        state = AutoMemoryWrapperState(messages=[])
        result = await wrapper_graph.capture_user_message_node(state)

        assert result == state
        mock_memory_editor.add_items.assert_not_called()

    async def test_memory_retrieve_node(self, wrapper_graph, mock_memory_editor):
        """Test memory_retrieve_node retrieves and injects memory."""
        mock_memory_editor.search.return_value = [
            MemoryItem(conversation=[], memory="Previous context 1", user_id="default_user"),
            MemoryItem(conversation=[], memory="Previous context 2", user_id="default_user")
        ]

        state = AutoMemoryWrapperState(messages=[HumanMessage(content="What did I say before?")])
        result = await wrapper_graph.memory_retrieve_node(state)

        mock_memory_editor.search.assert_called_once()
        # Memory should be inserted before the last user message
        assert len(result.messages) == 2
        assert isinstance(result.messages[0], SystemMessage)
        assert "Previous context 1" in result.messages[0].content
        assert "Previous context 2" in result.messages[0].content
        assert isinstance(result.messages[1], HumanMessage)

    async def test_memory_retrieve_node_no_results(self, wrapper_graph, mock_memory_editor):
        """Test memory_retrieve_node when no memories are found."""
        mock_memory_editor.search.return_value = []

        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Test")])
        result = await wrapper_graph.memory_retrieve_node(state)

        mock_memory_editor.search.assert_called_once()
        # No memory message should be added
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], HumanMessage)

    async def test_memory_retrieve_node_disabled(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test memory_retrieve_node when disabled."""
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=False,
                                             retrieve_memory=False,
                                             save_ai_responses=False)

        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Test")])
        result = await wrapper.memory_retrieve_node(state)

        assert result == state
        mock_memory_editor.search.assert_not_called()

    async def test_inner_agent_node(self, wrapper_graph, mock_inner_agent):
        """Test inner_agent_node calls inner agent and adds response."""
        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Calculate 2+2")])
        result = await wrapper_graph.inner_agent_node(state)

        # Verify inner agent was called
        mock_inner_agent.ainvoke.assert_called_once()
        call_args = mock_inner_agent.ainvoke.call_args
        chat_request = call_args[0][0]
        assert isinstance(chat_request, ChatRequest)
        assert len(chat_request.messages) == 1

        # Verify AI response was added
        assert len(result.messages) == 2
        assert isinstance(result.messages[1], AIMessage)
        assert result.messages[1].content == "Agent response"

    async def test_inner_agent_node_with_memory_context(self, wrapper_graph, mock_inner_agent):
        """Test inner_agent_node passes memory context to inner agent."""
        state = AutoMemoryWrapperState(
            messages=[SystemMessage(content="Memory context"), HumanMessage(content="User query")])
        await wrapper_graph.inner_agent_node(state)

        # Verify inner agent received both messages
        call_args = mock_inner_agent.ainvoke.call_args
        chat_request = call_args[0][0]
        assert len(chat_request.messages) == 2
        assert chat_request.messages[0].role == UserMessageContentRoleType.SYSTEM
        assert chat_request.messages[1].role == UserMessageContentRoleType.USER

    async def test_capture_ai_response_node(self, wrapper_graph, mock_memory_editor):
        """Test capture_ai_response_node saves AI responses."""
        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Question"), AIMessage(content="Answer")])
        result = await wrapper_graph.capture_ai_response_node(state)

        assert result == state
        mock_memory_editor.add_items.assert_called_once()
        call_args = mock_memory_editor.add_items.call_args
        items = call_args[0][0]
        assert len(items) == 1
        assert items[0].conversation == [{"role": "assistant", "content": "Answer"}]

    async def test_capture_ai_response_node_disabled(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test capture_ai_response_node when disabled."""
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=False,
                                             retrieve_memory=False,
                                             save_ai_responses=False)

        state = AutoMemoryWrapperState(messages=[AIMessage(content="Response")])
        result = await wrapper.capture_ai_response_node(state)

        assert result == state
        mock_memory_editor.add_items.assert_not_called()

    def test_build_graph_all_features(self, wrapper_graph):
        """Test build_graph creates workflow with all features enabled."""
        graph = wrapper_graph.build_graph()
        assert graph is not None

    def test_build_graph_minimal_features(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test build_graph with minimal features."""
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=False,
                                             retrieve_memory=False,
                                             save_ai_responses=False)
        graph = wrapper.build_graph()
        assert graph is not None

    async def test_search_params_passed_to_memory(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test that search_params are properly passed to memory editor."""
        search_params = {"top_k": 10, "mode": "summary"}
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=False,
                                             retrieve_memory=True,
                                             save_ai_responses=False,
                                             search_params=search_params)

        mock_memory_editor.search.return_value = []
        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Test")])
        await wrapper.memory_retrieve_node(state)

        call_kwargs = mock_memory_editor.search.call_args[1]
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["mode"] == "summary"

    async def test_add_params_passed_to_memory(self, mock_inner_agent, mock_memory_editor, mock_context):
        """Test that add_params are properly passed to memory editor."""
        add_params = {"ignore_roles": ["assistant"]}
        with patch('nat.agent.auto_memory_wrapper.agent.Context.get', return_value=mock_context):
            wrapper = AutoMemoryWrapperGraph(inner_agent_fn=mock_inner_agent,
                                             memory_editor=mock_memory_editor,
                                             save_user_messages=True,
                                             retrieve_memory=False,
                                             save_ai_responses=False,
                                             add_params=add_params)

        state = AutoMemoryWrapperState(messages=[HumanMessage(content="Test")])
        await wrapper.capture_user_message_node(state)

        call_kwargs = mock_memory_editor.add_items.call_args[1]
        assert call_kwargs["ignore_roles"] == ["assistant"]
