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
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig

NEMOTRON = "nvidia/nvidia-nemotron-nano-9b-v2"


def _fake_litellm_acompletion(fake):
    """Patch the litellm call under ADK's LiteLLMClient (ADK imports litellm lazily, so import it first)."""
    from google.adk.models import lite_llm

    lite_llm._ensure_litellm_imported()
    return patch.object(lite_llm, "acompletion", fake)


def _request(llm):
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    return LlmRequest(model=llm.model, contents=[types.Content(role="user", parts=[types.Part(text="hi")])])


def _rate_limit(model: str, message: str = "429 Too Many Requests"):
    import litellm

    return litellm.RateLimitError(message=message, llm_provider="openai", model=model)


async def _generate(config,
                    fail_first: int = 0,
                    error=_rate_limit,
                    request=None) -> tuple[list[list[dict]], str | None, Exception | None]:
    """
    Run one ADK request through a real WorkflowBuilder and LiteLlm.

    Only litellm's `acompletion` (the network call under ADK's LiteLLMClient) is faked.
    """
    import litellm

    import nat.plugins.adk.llm  # noqa: F401  registers the ADK clients

    sent: list[list[dict]] = []

    async def fake_acompletion(model, messages, tools=None, **kwargs):
        sent.append(messages)
        if len(sent) <= fail_first:
            raise error(model)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hello"}}])

    text, raised = None, None
    async with WorkflowBuilder() as builder:
        await builder.add_llm("llm", config)
        llm = await builder.get_llm("llm", wrapper_type=LLMFrameworkEnum.ADK)
        with _fake_litellm_acompletion(fake_acompletion), \
             patch("asyncio.sleep"):
            try:
                async for response in llm.generate_content_async(request(llm) if request else _request(llm)):
                    text = response.content.parts[0].text
            except Exception as e:  # noqa: BLE001 the test inspects it
                raised = e
    return sent, text, raised


# retry_on_errors=[] so only the status code can trigger the retry
@pytest.mark.parametrize(
    "config",
    [
        OpenAIModelConfig(model_name="gpt-4o-mini", api_key="test", retry_on_errors=[]),
        NIMModelConfig(model_name=NEMOTRON, api_key="test", retry_on_errors=[]),
        LiteLlmModelConfig(model_name="openai/gpt-4o-mini", api_key="test", retry_on_errors=[]),
        AzureOpenAIModelConfig(azure_deployment="gpt-4o-mini",
                               azure_endpoint="https://example.invalid",
                               api_key="test",
                               retry_on_errors=[]),
        DynamoModelConfig(model_name="llama", base_url="http://localhost:8000/v1", retry_on_errors=[]),
    ],
    ids=["openai", "nim", "litellm", "azure_openai", "dynamo"])
async def test_adk_client_retries_a_rate_limit(config):
    sent, text, error = await _generate(config, fail_first=1, error=lambda model: _rate_limit(model, "slow down"))
    assert error is None
    assert len(sent) == 2
    assert text == "hello"


async def test_adk_client_retries_on_a_configured_error_message():
    import litellm

    def backend_busy(model):
        return litellm.APIConnectionError(message="backend busy", llm_provider="openai", model=model)

    config = OpenAIModelConfig(model_name="gpt-4o-mini",
                               api_key="test",
                               retry_on_status_codes=[],
                               retry_on_errors=["backend busy"])
    sent, text, error = await _generate(config, fail_first=1, error=backend_busy)
    assert error is None
    assert len(sent) == 2


async def test_thinking_prompt_works_with_tools_that_cannot_be_copied():
    """ADK keeps tool objects in the request; an MCP tool holds a lock, which can't be deep-copied."""
    import threading

    class ToolWithLock:

        def __init__(self):
            self.lock = threading.Lock()

    def request_with_tool(llm):
        request = _request(llm)
        request.tools_dict = {"mcp_tool": ToolWithLock()}
        return request

    sent, _, error = await _generate(NIMModelConfig(model_name=NEMOTRON, api_key="test", thinking=True),
                                     request=request_with_tool)
    assert error is None
    assert "/think" in [m for m in sent[0] if m["role"] == "system"][0]["content"]


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

    async def fake_acompletion(model, messages, tools=None, **kwargs):
        sent.append(messages)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hello"}}])

    async with WorkflowBuilder() as builder:
        await builder.add_llm("llm", NIMModelConfig(model_name=NEMOTRON, api_key="test", thinking=True))
        llm = await builder.get_llm("llm", wrapper_type=LLMFrameworkEnum.ADK)
        request = _request(llm)
        request.config = types.GenerateContentConfig(system_instruction="You answer billing questions.")
        with _fake_litellm_acompletion(fake_acompletion):
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
