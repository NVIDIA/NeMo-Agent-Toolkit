# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable
from typing import Annotated
from typing import Any
from typing import TypedDict
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import AgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Message
from nat.data_models.api_server import UserMessageContentRoleType
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperConfig
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperFunction
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperInput
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperOutput
from nat.plugins.langchain.langgraph_workflow import split_graph_path


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        ("/path/to/module.py:graph_name", ("/path/to/module.py", "graph_name")),
        (r"C:\\path\\to\\module.py:graph", (r"C:\\path\\to\\module.py", "graph")),
        ("relative.py:graph", ("relative.py", "graph")),
    ],
)
def test_split_graph_path(graph, expected):
    """Graph references split at the final colon, including Windows drive paths."""
    assert split_graph_path(graph) == expected


@pytest.mark.parametrize("graph", ["module.py", ":graph", "module.py:", ""])
def test_split_graph_path_rejects_missing_parts(graph):
    """Graph references require both a module path and graph name."""
    with pytest.raises(ValueError, match="non-empty module path"):
        split_graph_path(graph)


class TestConvertChatRequest:
    """Tests for LanggraphWrapperFunction.convert_chat_request: ChatRequest → LanggraphWrapperInput."""

    def test_single_user_message(self):
        """Test converting a single user message."""
        chat_req = ChatRequest(messages=[Message(content="hello", role=UserMessageContentRoleType.USER)])
        result = LanggraphWrapperFunction.convert_chat_request(chat_req)

        assert isinstance(result, LanggraphWrapperInput)
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], HumanMessage)
        assert result.messages[0].content == "hello"

    def test_multi_turn(self):
        """Test converting a multi-turn conversation."""
        chat_req = ChatRequest(messages=[
            Message(content="hello", role=UserMessageContentRoleType.USER),
            Message(content="hi there", role=UserMessageContentRoleType.ASSISTANT),
            Message(content="how are you?", role=UserMessageContentRoleType.USER),
        ])
        result = LanggraphWrapperFunction.convert_chat_request(chat_req)

        assert isinstance(result, LanggraphWrapperInput)
        assert len(result.messages) == 3
        assert isinstance(result.messages[0], HumanMessage)
        assert isinstance(result.messages[1], AIMessage)
        assert isinstance(result.messages[2], HumanMessage)
        assert result.messages[2].content == "how are you?"

    def test_system_message(self):
        """Test converting a system message."""
        chat_req = ChatRequest(messages=[
            Message(content="You are helpful.", role=UserMessageContentRoleType.SYSTEM),
            Message(content="hello", role=UserMessageContentRoleType.USER),
        ])
        result = LanggraphWrapperFunction.convert_chat_request(chat_req)

        assert len(result.messages) == 2
        assert isinstance(result.messages[0], SystemMessage)
        assert result.messages[0].content == "You are helpful."
        assert isinstance(result.messages[1], HumanMessage)


class TestConvertStr:
    """Tests for LanggraphWrapperFunction.convert_str: str → LanggraphWrapperInput."""

    def test_plain_text(self):
        """Test converting a plain text string."""
        result = LanggraphWrapperFunction.convert_str("hello")

        assert isinstance(result, LanggraphWrapperInput)
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], HumanMessage)
        assert result.messages[0].content == "hello"

    def test_empty_string(self):
        """Test converting an empty string."""
        result = LanggraphWrapperFunction.convert_str("")

        assert isinstance(result, LanggraphWrapperInput)
        assert len(result.messages) == 1
        assert result.messages[0].content == ""


class TestConvertOutputToChatResponse:
    """Tests for LanggraphWrapperFunction.convert_to_chat_response: LanggraphWrapperOutput → ChatResponse."""

    def test_single_ai_message(self):
        """Test converting output with a single AI message."""
        output = LanggraphWrapperOutput(messages=[AIMessage(content="Echo: hello")])
        result = LanggraphWrapperFunction.convert_to_chat_response(output)

        assert isinstance(result, ChatResponse)
        assert result.choices[0].message.content == "Echo: hello"
        assert result.object == "chat.completion"

    def test_empty_messages(self):
        """Test converting output with no messages."""
        output = LanggraphWrapperOutput(messages=[])
        result = LanggraphWrapperFunction.convert_to_chat_response(output)

        assert isinstance(result, ChatResponse)
        assert result.choices[0].message.content == ""

    def test_multi_message_uses_last(self):
        """Test that the last message content is used."""
        output = LanggraphWrapperOutput(messages=[AIMessage(content="first"), AIMessage(content="second")])
        result = LanggraphWrapperFunction.convert_to_chat_response(output)

        assert result.choices[0].message.content == "second"


class TestConvertOutputToChatResponseChunk:
    """Tests for LanggraphWrapperFunction.convert_to_chat_response_chunk: LanggraphWrapperOutput → ChatResponseChunk."""

    def test_single_ai_message(self):
        """Test converting output with a single AI message."""
        output = LanggraphWrapperOutput(messages=[AIMessage(content="Echo: hello")])
        result = LanggraphWrapperFunction.convert_to_chat_response_chunk(output)

        assert isinstance(result, ChatResponseChunk)
        assert result.choices[0].delta.content == "Echo: hello"
        assert result.object == "chat.completion.chunk"

    def test_empty_messages(self):
        """Test converting output with no messages."""
        output = LanggraphWrapperOutput(messages=[])
        result = LanggraphWrapperFunction.convert_to_chat_response_chunk(output)

        assert isinstance(result, ChatResponseChunk)
        assert result.choices[0].delta.content == ""

    def test_multi_message_uses_last(self):
        """Test that the last message content is used."""
        output = LanggraphWrapperOutput(messages=[AIMessage(content="first"), AIMessage(content="second")])
        result = LanggraphWrapperFunction.convert_to_chat_response_chunk(output)

        assert result.choices[0].delta.content == "second"


class TestConvertToStr:
    """Tests for LanggraphWrapperFunction.convert_to_str: LanggraphWrapperOutput → str."""

    def test_single_ai_message(self):
        """Test extracting content from a single AI message."""
        output = LanggraphWrapperOutput(messages=[AIMessage(content="Echo: hello")])
        result = LanggraphWrapperFunction.convert_to_str(output)
        assert result == "Echo: hello"

    def test_empty_messages(self):
        """Test extracting content when no messages are present."""
        output = LanggraphWrapperOutput(messages=[])
        result = LanggraphWrapperFunction.convert_to_str(output)
        assert result == ""

    def test_multi_message_returns_last(self):
        """Test that the last message content is returned."""
        output = LanggraphWrapperOutput(messages=[AIMessage(content="first"), AIMessage(content="second")])
        result = LanggraphWrapperFunction.convert_to_str(output)
        assert result == "second"


class TestParseStreamOutput:
    """Tests for LanggraphWrapperFunction._parse_stream_output."""

    def test_flat_dict(self):
        """Test parsing a flat messages dict."""
        raw = {"messages": [AIMessage(content="hi")]}
        result = LanggraphWrapperFunction._parse_stream_output(raw)

        assert isinstance(result, LanggraphWrapperOutput)
        assert result.messages[0].content == "hi"

    def test_node_keyed_dict(self):
        """Test parsing a single-node-keyed dict from LangGraph astream."""
        raw = {"echo": {"messages": [AIMessage(content="hi")]}}
        result = LanggraphWrapperFunction._parse_stream_output(raw)

        assert isinstance(result, LanggraphWrapperOutput)
        assert result.messages[0].content == "hi"

    def test_multi_key_dict_raises(self):
        """Test that multiple node keys raises an error."""
        raw = {
            "node_a": {
                "messages": [AIMessage(content="a")]
            },
            "node_b": {
                "messages": [AIMessage(content="b")]
            },
        }
        with pytest.raises(Exception):
            LanggraphWrapperFunction._parse_stream_output(raw)

    def test_node_update_without_messages_is_skipped(self):
        """A node that only updates other state keys carries nothing to stream."""
        raw = {"classify": {"route": "faq"}}

        assert LanggraphWrapperFunction._parse_stream_output(raw) is None

    def test_node_returning_none_is_skipped(self):
        """A node that returns nothing, such as an agent middleware hook, carries nothing to stream."""
        raw = {"audit": None}

        assert LanggraphWrapperFunction._parse_stream_output(raw) is None

    def test_single_message_update(self):
        """A node may return one message instead of a list, which the add_messages reducer accepts."""
        raw = {"answer": {"messages": AIMessage(content="hi")}}
        result = LanggraphWrapperFunction._parse_stream_output(raw)

        assert isinstance(result, LanggraphWrapperOutput)
        assert [message.content for message in result.messages] == ["hi"]

    def test_tuple_of_messages_update(self):
        """A node may return its messages as a tuple, which is parsed as a collection and not as one message."""
        raw = {"answer": {"messages": (AIMessage(content="hi"), AIMessage(content="there"))}}
        result = LanggraphWrapperFunction._parse_stream_output(raw)

        assert isinstance(result, LanggraphWrapperOutput)
        assert [message.content for message in result.messages] == ["hi", "there"]

    def test_invalid_message_in_list_raises(self):
        """A list that holds something other than a message is still rejected."""
        raw = {"answer": {"messages": [object()]}}

        with pytest.raises(ValueError, match="messages.0"):
            LanggraphWrapperFunction._parse_stream_output(raw)


class _GraphState(TypedDict, total=False):
    """State with a messages channel and one unrelated key."""

    messages: Annotated[list[AnyMessage], add_messages]
    route: str


def _answer(_state: _GraphState) -> dict[str, Any]:
    return {"messages": [AIMessage(content="hello")]}


def _graph_with_first_node(first_node: Callable[[_GraphState], Any]) -> CompiledStateGraph:
    """Compile a graph that runs first_node and then a node that answers with "hello"."""
    builder = StateGraph(_GraphState)
    builder.add_node("first", first_node)
    builder.add_node("answer", _answer)
    builder.add_edge(START, "first")
    builder.add_edge("first", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def _wrap(graph: CompiledStateGraph) -> LanggraphWrapperFunction:
    return LanggraphWrapperFunction(config=LanggraphWrapperConfig(graph="module.py:graph"), graph=graph)


class _NoOpMiddleware(AgentMiddleware):
    """Agent middleware whose hook leaves the state unchanged."""

    def before_model(self, state: AgentState, runtime: Runtime) -> None:
        return None


class TestStreamCompiledGraph:
    """Tests for streaming compiled graphs whose nodes do not all return a list of messages."""

    @pytest.mark.parametrize(
        "first_node",
        [
            pytest.param(lambda _state: {"route": "faq"}, id="updates_other_state_key"),
            pytest.param(lambda _state: None, id="returns_none"),
        ],
    )
    async def test_node_without_messages_does_not_end_the_stream(self, first_node: Callable[[_GraphState], Any]):
        """Streaming returns the same answer as ainvoke when a node update has no messages."""
        function = _wrap(_graph_with_first_node(first_node))

        assert await function.ainvoke("hi", to_type=str) == "hello"
        assert [chunk async for chunk in function.astream("hi", to_type=str)] == ["hello"]

    async def test_node_returning_a_single_message(self):
        """Streaming accepts a node that returns one message instead of a list."""
        builder = StateGraph(_GraphState)
        builder.add_node("answer", lambda _state: {"messages": AIMessage(content="hello")})
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        function = _wrap(builder.compile())

        assert await function.ainvoke("hi", to_type=str) == "hello"
        assert [chunk async for chunk in function.astream("hi", to_type=str)] == ["hello"]

    async def test_create_agent_with_middleware_that_returns_none(self):
        """Streaming works for a LangChain agent whose middleware hook returns None."""

        def build_agent() -> CompiledStateGraph:
            model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))
            return create_agent(model=model, tools=[], middleware=[_NoOpMiddleware()])

        assert await _wrap(build_agent()).ainvoke("hi", to_type=str) == "hello"
        assert [chunk async for chunk in _wrap(build_agent()).astream("hi", to_type=str)] == ["hello"]


class TestBuildRunnableConfig:
    """Tests for LanggraphWrapperFunction._build_runnable_config: conversation_id -> LangGraph thread_id."""

    def test_conversation_id_maps_to_thread_id(self):
        """When a conversation_id is in context, it is passed as the LangGraph thread_id."""
        with patch("nat.plugins.langchain.langgraph_workflow.Context") as mock_context:
            mock_context.get.return_value.conversation_id = "conv-123"

            config = LanggraphWrapperFunction._build_runnable_config()

        assert config is not None
        assert config["configurable"]["thread_id"] == "conv-123"

    def test_no_conversation_id_returns_none(self):
        """When no conversation_id is in context, no config is built (prior behavior preserved)."""
        with patch("nat.plugins.langchain.langgraph_workflow.Context") as mock_context:
            mock_context.get.return_value.conversation_id = None

            config = LanggraphWrapperFunction._build_runnable_config()

        assert config is None
