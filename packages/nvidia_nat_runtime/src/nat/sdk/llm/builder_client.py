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
"""LLM client adapter backed by a core RuntimeOpenAIClient from the Builder."""

from __future__ import annotations

import json
import logging
from typing import Any

from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import LLMResponse
from nat.sdk.llm.message import Message
from nat.sdk.llm.message import TokenUsage
from nat.sdk.llm.message import ToolCall
from nat.sdk.tool.tool import Tool

logger = logging.getLogger(__name__)


class BuilderLLMClient(LLMClient):
    """LLM client backed by a core :class:`RuntimeOpenAIClient` from the Builder.

    This adapter wraps a ``RuntimeOpenAIClient`` (obtained via the core Builder
    system) and inherits from :class:`LLMClient`, bridging the core runtime
    infrastructure with the lightweight SDK agent loop.
    """

    def __init__(self, runtime_client: Any) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        runtime_client
            A ``RuntimeOpenAIClient`` instance obtained from the Builder.
        """
        self._client = runtime_client

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the chat completions endpoint via the wrapped runtime client.

        Converts SDK :class:`Message` objects to OpenAI dicts, delegates to
        ``RuntimeOpenAIClient.chat_completions``, and parses the response back
        into an SDK :class:`LLMResponse`.
        """
        openai_messages = [m.to_openai_dict() for m in messages]

        extra_kwargs: dict[str, Any] = dict(kwargs)
        if tools is not None and tools:
            extra_kwargs["tools"] = [t.to_openai_schema() for t in tools]

        response = await self._client.chat_completions(openai_messages, **extra_kwargs)

        choice = response.choices[0]

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        message = Message(
            role="assistant",
            content=choice.message.content or "",
            tool_calls=tool_calls,
        )

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            message=message,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason,
            raw=response,
        )


async def create_llm_client(
    config: Any,
    *,
    llm_name: str = "default",
) -> BuilderLLMClient:
    """Create an :class:`LLMClient` from any core ``LLMBaseConfig``.

    This factory registers the config with the Builder, retrieves the
    ``RUNTIME`` wrapper (a ``RuntimeOpenAIClient``), and returns a
    :class:`BuilderLLMClient` adapter.

    Parameters
    ----------
    config
        A core ``LLMBaseConfig`` subclass instance (e.g. ``OpenAIModelConfig``,
        ``NIMModelConfig``).
    llm_name
        Name to register the LLM under in the builder. Defaults to
        ``"default"``.

    Returns
    -------
    BuilderLLMClient
        An SDK-compatible LLM client backed by the core runtime.
    """
    from nat.builder.builder import Builder

    builder = Builder.current()

    from nat.builder.framework_enum import LLMFrameworkEnum

    await builder.add_llm(llm_name, config)
    runtime_client = await builder.get_llm(llm_name, LLMFrameworkEnum.RUNTIME)
    return BuilderLLMClient(runtime_client)
