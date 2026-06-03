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

from unittest.mock import MagicMock

import pytest

from nat.builder.builder import Builder
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.function import EmptyFunctionConfig
from nat.plugins.langchain.tool_wrapper import langchain_tool_wrapper


def _content_text(content: object) -> str:
    if isinstance(content, list):
        first_content = content[0]
        if isinstance(first_content, dict):
            return first_content["text"]
        return str(first_content.text)
    return str(content)


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_maps_string_to_input_message() -> None:

    async def _echo(chat_request_or_message: ChatRequestOrMessage) -> str:
        if chat_request_or_message.input_message is not None:
            return chat_request_or_message.input_message
        return _content_text(chat_request_or_message.messages[-1].content)  # type: ignore[index, union-attr]

    info = FunctionInfo.from_fn(_echo, description="Echo input")
    fn = LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="echo")
    tool = langchain_tool_wrapper("echo", fn, MagicMock(spec=Builder))

    assert await tool.ainvoke("hello") == "hello"
    assert await tool.ainvoke({"messages": [{"role": "user", "content": "hi"}]}) == "hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": "json hi"}]'}) == "json hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": {"type": "text", "text": "json object hi"}}]'}
                              ) == "json object hi"


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_maps_string_to_chat_request() -> None:

    async def _echo(chat_request: ChatRequest) -> str:
        return _content_text(chat_request.messages[-1].content)

    info = FunctionInfo.from_fn(_echo, description="Echo input")
    fn = LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="echo")
    tool = langchain_tool_wrapper("echo", fn, MagicMock(spec=Builder))

    assert await tool.ainvoke("hello") == "hello"
    assert await tool.ainvoke({"messages": [{"role": "user", "content": "hi"}]}) == "hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": "json hi"}]'}) == "json hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": {"type": "text", "text": "json object hi"}}]'}
                              ) == "json object hi"
