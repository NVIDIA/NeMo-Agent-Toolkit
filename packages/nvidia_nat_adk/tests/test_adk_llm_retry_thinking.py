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
"""ADK clients honour the retry and thinking options of the LLM config, like the other framework wrappers."""

from unittest.mock import patch

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig

NEMOTRON = "nvidia/nvidia-nemotron-nano-9b-v2"


def _request(llm):
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    return LlmRequest(model=llm.model, contents=[types.Content(role="user", parts=[types.Part(text="hi")])])


async def _generate(config, fail_first: int = 0) -> tuple[list[list[dict]], str | None, Exception | None]:
    """Run one ADK request through a real WorkflowBuilder and LiteLlm; only the litellm network call is faked."""
    import litellm

    import nat.plugins.adk.llm  # noqa: F401  registers the ADK clients

    sent: list[list[dict]] = []

    async def fake_acompletion(self, model, messages, tools=None, **kwargs):
        sent.append(messages)
        if len(sent) <= fail_first:
            raise litellm.RateLimitError(message="429 Too Many Requests", llm_provider="openai", model=model)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hello"}}])

    text, error = None, None
    async with WorkflowBuilder() as builder:
        await builder.add_llm("llm", config)
        llm = await builder.get_llm("llm", wrapper_type=LLMFrameworkEnum.ADK)
        with patch("google.adk.models.lite_llm.LiteLLMClient.acompletion", fake_acompletion), \
             patch("asyncio.sleep"):
            try:
                async for response in llm.generate_content_async(_request(llm)):
                    text = response.content.parts[0].text
            except Exception as e:  # noqa: BLE001 the test inspects it
                error = e
    return sent, text, error


@pytest.mark.parametrize("config",
                         [
                             OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test"),
                             NIMModelConfig(model_name=NEMOTRON, api_key="test"),
                             LiteLlmModelConfig(model_name="openai/gpt-4o-mini", api_key="test"),
                         ],
                         ids=["openai", "nim", "litellm"])
async def test_adk_client_retries_a_rate_limit(config):
    sent, text, error = await _generate(config, fail_first=1)
    assert error is None
    assert len(sent) == 2
    assert text == "hello"


async def test_adk_client_stops_after_num_retries():
    sent, _, error = await _generate(OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", num_retries=3),
                                     fail_first=10)
    assert type(error).__name__ == "RateLimitError"
    assert len(sent) == 3


async def test_adk_client_does_not_retry_when_auto_retry_is_off():
    sent, _, error = await _generate(OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", do_auto_retry=False),
                                     fail_first=1)
    assert type(error).__name__ == "RateLimitError"
    assert len(sent) == 1


@pytest.mark.parametrize("thinking, prompt", [(True, "/think"), (False, "/no_think")])
async def test_adk_client_sends_the_thinking_system_prompt(thinking, prompt):
    sent, _, error = await _generate(NIMModelConfig(model_name=NEMOTRON, api_key="test", thinking=thinking))
    assert error is None
    system = [m for m in sent[0] if m["role"] == "system"]
    assert len(system) == 1
    assert prompt in system[0]["content"]


async def test_adk_client_keeps_the_agent_instruction_with_the_thinking_prompt():
    """An ADK agent's instruction arrives as the request's system_instruction; both reach the model."""
    import litellm
    from google.genai import types

    import nat.plugins.adk.llm  # noqa: F401  registers the ADK clients

    sent: list[list[dict]] = []

    async def fake_acompletion(self, model, messages, tools=None, **kwargs):
        sent.append(messages)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hello"}}])

    async with WorkflowBuilder() as builder:
        await builder.add_llm("llm", NIMModelConfig(model_name=NEMOTRON, api_key="test", thinking=True))
        llm = await builder.get_llm("llm", wrapper_type=LLMFrameworkEnum.ADK)
        request = _request(llm)
        request.config = types.GenerateContentConfig(system_instruction="You answer billing questions.")
        with patch("google.adk.models.lite_llm.LiteLLMClient.acompletion", fake_acompletion):
            async for _ in llm.generate_content_async(request):
                pass

    system = [m["content"] for m in sent[0] if m["role"] == "system"]
    assert len(system) == 1
    assert "You answer billing questions." in system[0] and "/think" in system[0]
    assert request.config.system_instruction == "You answer billing questions."  # the caller's request is unchanged


async def test_adk_client_sends_no_thinking_prompt_when_thinking_is_unset():
    sent, _, error = await _generate(NIMModelConfig(model_name=NEMOTRON, api_key="test"))
    assert error is None
    assert not [m for m in sent[0] if m["role"] == "system"]
