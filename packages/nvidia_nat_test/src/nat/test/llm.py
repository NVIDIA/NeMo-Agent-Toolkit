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

# pylint: disable=unused-argument,missing-class-docstring,missing-function-docstring,import-outside-toplevel,too-few-public-methods

import asyncio
import time
from collections.abc import AsyncGenerator
from collections.abc import Iterator
from itertools import cycle as iter_cycle
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.llm import LLMBaseConfig


class TestLLMConfig(LLMBaseConfig, name="nat_test_llm"):
    """Test LLM configuration."""
    __test__ = False
    response_seq: list[str] = Field(
        default=[],
        description=("returns the next element in order (wraps)"),
    )
    delay_ms: int = Field(default=0, ge=0, description="Artificial per-call delay in milliseconds to mimic latency")


class _ResponseChooser:
    """
    Helper class to choose the next response according to config using itertools.cycle and provide synchronous
    and asynchronous sleep functions.
    """

    def __init__(self, response_seq: list[str], delay_ms: int):
        self._cycler = iter_cycle(response_seq) if response_seq else None
        self._delay_ms = delay_ms

    def next_response(self) -> str:
        """Return the next response in the cycle, or an empty string if no responses are configured."""
        if self._cycler is None:
            return ""
        return next(self._cycler)

    def sync_sleep(self) -> None:
        time.sleep(self._delay_ms / 1000.0)

    async def async_sleep(self) -> None:
        await asyncio.sleep(self._delay_ms / 1000.0)


@register_llm_provider(config_type=TestLLMConfig)
async def test_llm_provider(config: TestLLMConfig, builder: Builder):
    del builder  # suppress linting error
    yield LLMProviderInfo(config=config, description="Test LLM provider")


@register_llm_client(config_type=TestLLMConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def test_llm_langchain(config: TestLLMConfig, builder: Builder):
    """LLM client for LangChain"""
    from langchain_core.messages import AIMessage

    del builder  # suppress linting error
    chooser = _ResponseChooser(response_seq=config.response_seq, delay_ms=config.delay_ms)

    class LangChainTestLLM:

        def __init__(self, config: TestLLMConfig) -> None:
            self.config = config

        def invoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
            chooser.sync_sleep()
            return AIMessage(content=chooser.next_response())

        async def ainvoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
            await chooser.async_sleep()
            return AIMessage(content=chooser.next_response())

        def stream(self, messages: Any, **_kwargs: Any) -> Iterator[AIMessage]:
            chooser.sync_sleep()
            yield AIMessage(content=chooser.next_response())

        async def astream(self, messages: Any, **_kwargs: Any) -> AsyncGenerator[AIMessage, None]:
            await chooser.async_sleep()
            yield AIMessage(content=chooser.next_response())

    yield LangChainTestLLM(config)


@register_llm_client(config_type=TestLLMConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def test_llm_llama_index(config: TestLLMConfig, builder: Builder):
    """LLM client for LlamaIndex"""
    del builder  # suppress linting error
    chooser = _ResponseChooser(response_seq=config.response_seq, delay_ms=config.delay_ms)

    class LIChatMessage:

        def __init__(self, content: str) -> None:
            self.content = content

    class LIChatResponse:

        def __init__(self, text: str) -> None:
            self.message = LIChatMessage(text)
            self.text = text

    class LITestLLM:

        def chat(self, messages: list[Any] | None = None, **_kwargs: Any) -> LIChatResponse:
            chooser.sync_sleep()
            return LIChatResponse(chooser.next_response())

        async def achat(self, messages: list[Any] | None = None, **_kwargs: Any) -> LIChatResponse:
            await chooser.async_sleep()
            return LIChatResponse(chooser.next_response())

        def stream_chat(self, messages: list[Any] | None = None, **_kwargs: Any) -> Iterator[LIChatResponse]:
            chooser.sync_sleep()
            yield LIChatResponse(chooser.next_response())

        async def astream_chat(self,
                               messages: list[Any] | None = None,
                               **_kwargs: Any) -> AsyncGenerator[LIChatResponse, None]:
            await chooser.async_sleep()
            yield LIChatResponse(chooser.next_response())

    yield LITestLLM()


@register_llm_client(config_type=TestLLMConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def test_llm_crewai(config: TestLLMConfig, builder: Builder):
    """LLM client for CrewAI"""
    del builder  # suppress linting error
    chooser = _ResponseChooser(response_seq=config.response_seq, delay_ms=config.delay_ms)

    class CrewAITestLLM:

        def call(self, messages: list[dict[str, str]] | None = None, **kwargs: Any) -> str:
            chooser.sync_sleep()
            return chooser.next_response()

    yield CrewAITestLLM()


@register_llm_client(config_type=TestLLMConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def test_llm_semantic_kernel(config: TestLLMConfig, builder: Builder):
    """LLM client for SemanticKernel"""
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole

    del builder  # suppress linting error
    chooser = _ResponseChooser(response_seq=config.response_seq, delay_ms=config.delay_ms)

    class SKTestLLM:

        async def get_chat_message_contents(self, chat_history: Any, **_kwargs: Any) -> list[ChatMessageContent]:
            await chooser.async_sleep()
            text = chooser.next_response()
            return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=text)]

        async def get_streaming_chat_message_contents(self, chat_history: Any,
                                                      **_kwargs: Any) -> AsyncGenerator[ChatMessageContent, None]:
            await chooser.async_sleep()
            text = chooser.next_response()
            yield ChatMessageContent(role=AuthorRole.ASSISTANT, content=text)

    yield SKTestLLM()


@register_llm_client(config_type=TestLLMConfig, wrapper_type=LLMFrameworkEnum.AGNO)
async def test_llm_agno(config: TestLLMConfig, builder: Builder):
    """LLM client for agno"""
    del builder  # suppress linting error
    chooser = _ResponseChooser(response_seq=config.response_seq, delay_ms=config.delay_ms)

    class AgnoTestLLM:

        def invoke(self, messages: Any | None = None, **_kwargs: Any) -> str:
            chooser.sync_sleep()
            return chooser.next_response()

        async def ainvoke(self, messages: Any | None = None, **_kwargs: Any) -> str:
            await chooser.async_sleep()
            return chooser.next_response()

        def invoke_stream(self, messages: Any | None = None, **_kwargs: Any) -> Iterator[str]:
            chooser.sync_sleep()
            yield chooser.next_response()

        async def ainvoke_stream(self, messages: Any | None = None, **_kwargs: Any) -> AsyncGenerator[str, None]:
            await chooser.async_sleep()
            yield chooser.next_response()

    yield AgnoTestLLM()
