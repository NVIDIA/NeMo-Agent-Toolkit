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
"""Semantic Kernel clients send the sampling options of the LLM config and use its max_retries."""

from unittest.mock import patch

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.openai_llm import OpenAIModelConfig


def _completion():
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{
            "index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "hello"
            }
        }],
    })


async def _agent_request(config, settings=None, fail_first: int = 0) -> tuple[list[dict], object]:
    """Ask a ChatCompletionAgent (as the SK demo does) through a NAT-built client; only the HTTP call is faked."""
    import openai
    from semantic_kernel import Kernel
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.functions import KernelArguments

    import nat.plugins.semantic_kernel.llm  # noqa: F401  registers the SK clients

    sent: list[dict] = []

    async def fake_create(self, **kwargs):
        sent.append(kwargs)
        if len(sent) <= fail_first:
            raise openai.RateLimitError("429 Too Many Requests", response=_rate_limit_response(), body=None)
        return _completion()

    async with WorkflowBuilder() as builder:
        await builder.add_llm("llm", config)
        llm = await builder.get_llm("llm", wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
        kernel = Kernel()
        kernel.add_service(llm)
        agent = ChatCompletionAgent(kernel=kernel,
                                    name="agent",
                                    instructions="be brief",
                                    arguments=KernelArguments(settings=settings) if settings else None)
        with patch("openai.resources.chat.completions.AsyncCompletions.create", fake_create), \
             patch("asyncio.sleep"):
            await agent.get_response(messages="hi")
    return sent, llm


def _rate_limit_response():
    import httpx

    return httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))


@pytest.mark.parametrize(
    "config",
    [
        OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", temperature=0.0, top_p=0.5, seed=7),
        AzureOpenAIModelConfig(azure_deployment="gpt-4o-mini",
                               azure_endpoint="https://example.invalid",
                               api_key="test",
                               temperature=0.0,
                               top_p=0.5,
                               seed=7),
    ],
    ids=["openai", "azure_openai"])
async def test_agent_request_carries_the_config_sampling_options(config):
    sent, _ = await _agent_request(config)
    assert (sent[0].get("temperature"), sent[0].get("top_p"), sent[0].get("seed")) == (0.0, 0.5, 7)


async def test_settings_passed_by_the_caller_win_over_the_config():
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

    config = OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", temperature=0.0, top_p=0.5, seed=7)
    sent, _ = await _agent_request(config, settings=OpenAIChatPromptExecutionSettings(temperature=0.9))
    assert (sent[0].get("temperature"), sent[0].get("top_p"), sent[0].get("seed")) == (0.9, 0.5, 7)


async def test_unset_sampling_options_are_not_sent():
    sent, _ = await _agent_request(OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", temperature=None,
                                                     top_p=None))
    assert sent[0].get("temperature") is None and sent[0].get("top_p") is None and sent[0].get("seed") is None


async def test_openai_client_uses_the_config_max_retries():
    _, llm = await _agent_request(OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", max_retries=0))
    assert llm.client.max_retries == 0


async def test_automatic_retries_still_apply_with_sampling_options():
    config = OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", temperature=0.0, max_retries=0)
    sent, _ = await _agent_request(config, fail_first=1)
    assert len(sent) == 2
    assert all(request.get("temperature") == 0.0 for request in sent)


async def test_streaming_agent_request_carries_the_config_sampling_options():
    from openai.types.chat import ChatCompletionChunk
    from semantic_kernel import Kernel
    from semantic_kernel.agents import ChatCompletionAgent

    import nat.plugins.semantic_kernel.llm  # noqa: F401  registers the SK clients

    sent: list[dict] = []

    chunk = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{
            "index": 0, "delta": {
                "role": "assistant", "content": "hello"
            }, "finish_reason": "stop"
        }],
    }

    async def fake_create(self, **kwargs):
        # SK only treats an openai.AsyncStream as a streamed reply
        import json

        import httpx
        from openai import AsyncStream

        sent.append(kwargs)
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
        response = httpx.Response(200,
                                  content=body,
                                  headers={"content-type": "text/event-stream"},
                                  request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))
        return AsyncStream(cast_to=ChatCompletionChunk, response=response, client=self._client)

    config = OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", temperature=0.0, top_p=0.5, seed=7)
    async with WorkflowBuilder() as builder:
        await builder.add_llm("llm", config)
        llm = await builder.get_llm("llm", wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
        kernel = Kernel()
        kernel.add_service(llm)
        agent = ChatCompletionAgent(kernel=kernel, name="agent", instructions="be brief")
        with patch("openai.resources.chat.completions.AsyncCompletions.create", fake_create):
            text = "".join([str(chunk.content) async for chunk in agent.invoke_stream(messages="hi")])

    assert "hello" in text
    assert sent[0].get("stream") is True
    assert (sent[0].get("temperature"), sent[0].get("top_p"), sent[0].get("seed")) == (0.0, 0.5, 7)
