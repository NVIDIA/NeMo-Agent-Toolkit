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

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Message
from nat.data_models.api_server import UserMessageContentRoleType
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperFunction
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperInput
from nat.plugins.langchain.langgraph_workflow import LanggraphWrapperOutput


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
