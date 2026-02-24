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
"""Tests for nat.sdk.llm.builder_client — BuilderLLMClient and create_llm_client."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from nat.sdk.llm.builder_client import BuilderLLMClient
from nat.sdk.llm.builder_client import create_llm_client
from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import Message
from nat.sdk.tool.tool import Tool

# ---------------------------------------------------------------------------
# Helpers — fake OpenAI response objects
# ---------------------------------------------------------------------------


def _make_fake_response(
    content: str = "Hello!",
    tool_calls: list | None = None,
    model: str = "test-model",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> MagicMock:
    """Build a mock ChatCompletion response matching the OpenAI SDK shape."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


def _make_tool_call(tc_id: str, name: str, arguments: str) -> MagicMock:
    """Build a mock tool call matching the OpenAI SDK shape."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments

    tc = MagicMock()
    tc.id = tc_id
    tc.function = fn
    return tc


# ---------------------------------------------------------------------------
# BuilderLLMClient
# ---------------------------------------------------------------------------


class TestBuilderLLMClient:

    async def test_simple_completion(self) -> None:
        runtime_client = AsyncMock()
        runtime_client.chat_completions.return_value = _make_fake_response(content="Hi there")

        client = BuilderLLMClient(runtime_client)
        messages = [Message(role="user", content="Hello")]
        response = await client.complete(messages, [])

        assert response.message.role == "assistant"
        assert response.message.content == "Hi there"
        assert response.model == "test-model"
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    async def test_with_tool_calls(self) -> None:
        tc = _make_tool_call("tc-1", "add", '{"a": 1, "b": 2}')
        runtime_client = AsyncMock()
        runtime_client.chat_completions.return_value = _make_fake_response(content="", tool_calls=[tc])

        client = BuilderLLMClient(runtime_client)
        messages = [Message(role="user", content="Add 1+2")]
        response = await client.complete(messages, [])

        assert len(response.message.tool_calls) == 1
        assert response.message.tool_calls[0].name == "add"
        assert response.message.tool_calls[0].arguments == {"a": 1, "b": 2}
        assert response.message.tool_calls[0].id == "tc-1"

    async def test_malformed_tool_call_arguments(self) -> None:
        tc = _make_tool_call("tc-2", "broken", "not-valid-json")
        runtime_client = AsyncMock()
        runtime_client.chat_completions.return_value = _make_fake_response(content="", tool_calls=[tc])

        client = BuilderLLMClient(runtime_client)
        messages = [Message(role="user", content="test")]
        response = await client.complete(messages, [])

        assert len(response.message.tool_calls) == 1
        assert response.message.tool_calls[0].arguments == {"_raw": "not-valid-json"}

    async def test_tools_passed_to_runtime(self) -> None:
        runtime_client = AsyncMock()
        runtime_client.chat_completions.return_value = _make_fake_response()

        tool = Tool(
            name="search",
            description="Search for things",
            parameters={
                "type": "object", "properties": {
                    "q": {
                        "type": "string"
                    }
                }
            },
        )

        client = BuilderLLMClient(runtime_client)
        messages = [Message(role="user", content="test")]
        await client.complete(messages, [tool])

        # Verify tools were passed as OpenAI schema
        call_kwargs = runtime_client.chat_completions.call_args
        assert "tools" in call_kwargs.kwargs

    async def test_no_usage(self) -> None:
        response = _make_fake_response()
        response.usage = None

        runtime_client = AsyncMock()
        runtime_client.chat_completions.return_value = response

        client = BuilderLLMClient(runtime_client)
        messages = [Message(role="user", content="test")]
        result = await client.complete(messages, [])

        assert result.usage is None

    async def test_implements_base_class(self) -> None:
        runtime_client = AsyncMock()
        client = BuilderLLMClient(runtime_client)
        assert isinstance(client, LLMClient)

    async def test_messages_converted_to_openai_format(self) -> None:
        runtime_client = AsyncMock()
        runtime_client.chat_completions.return_value = _make_fake_response()

        client = BuilderLLMClient(runtime_client)
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hi"),
        ]
        await client.complete(messages, [])

        call_args = runtime_client.chat_completions.call_args
        openai_messages = call_args.args[0]
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are helpful"
        assert openai_messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# create_llm_client factory
# ---------------------------------------------------------------------------


class TestCreateLLMClient:

    async def test_factory_with_mock_builder(self) -> None:
        mock_runtime_client = AsyncMock()
        mock_runtime_client.chat_completions.return_value = _make_fake_response()

        mock_builder = AsyncMock()
        mock_builder.get_llm.return_value = mock_runtime_client

        mock_config = MagicMock()

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            client = await create_llm_client(
                mock_config,
                llm_name="test-llm",
            )

        assert isinstance(client, BuilderLLMClient)
        mock_builder.add_llm.assert_called_once_with("test-llm", mock_config)
        mock_builder.get_llm.assert_called_once()

    async def test_factory_default_name(self) -> None:
        mock_builder = AsyncMock()
        mock_builder.get_llm.return_value = AsyncMock()

        mock_config = MagicMock()

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            await create_llm_client(mock_config)

        mock_builder.add_llm.assert_called_once_with("default", mock_config)

    async def test_factory_returns_working_client(self) -> None:
        mock_runtime_client = AsyncMock()
        mock_runtime_client.chat_completions.return_value = _make_fake_response(content="Built!")

        mock_builder = AsyncMock()
        mock_builder.get_llm.return_value = mock_runtime_client

        mock_config = MagicMock()

        with patch("nat.builder.builder.Builder.current", return_value=mock_builder):
            client = await create_llm_client(mock_config)
        messages = [Message(role="user", content="test")]
        response = await client.complete(messages, [])

        assert response.message.content == "Built!"
