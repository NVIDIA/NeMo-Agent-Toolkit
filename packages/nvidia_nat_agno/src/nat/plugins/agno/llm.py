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

from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import APITypeEnum
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    from agno.models.message import Message

    class AgnoThinkingInjector(BaseThinkingInjector):

        from agno.models.message import Message

        @override
        def inject(self, messages: list[Message], *args, **kwargs) -> FunctionArgumentWrapper:
            # Attempt to inject the system prompt into the first system message
            for i, message in enumerate(messages):
                if message.role == "system":
                    if self.system_prompt not in str(message.content):
                        messages = list(messages)
                        messages[i] = Message(role="system", content=f"{message.content}\n{self.system_prompt}")
                    break
            else:
                messages = list(messages)
                messages.insert(0, Message(role="system", content=self.system_prompt))
            return FunctionArgumentWrapper(messages, *args, **kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            AgnoThinkingInjector(system_prompt=llm_config.thinking_system_prompt,
                                 function_names=[
                                     "invoke_stream",
                                     "invoke",
                                     "ainvoke",
                                     "ainvoke_stream",
                                 ]))

    return client


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.AGNO)
async def nim_agno(llm_config: NIMModelConfig, _builder: Builder):

    from agno.models.nvidia import Nvidia

    validate_no_responses_api(llm_config, LLMFrameworkEnum.AGNO)

    config_obj = {
        **llm_config.model_dump(
            exclude={"type", "model_name", "thinking", "api_type"},
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        ),
    }

    client = Nvidia(**config_obj, id=llm_config.model_name)

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AGNO)
async def openai_agno(llm_config: OpenAIModelConfig, _builder: Builder):

    from agno.models.openai import OpenAIChat
    from agno.models.openai import OpenAIResponses

    config_obj = {
        **llm_config.model_dump(
            exclude={"type", "model_name", "thinking", "api_type"},
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        ),
    }

    if llm_config.api_type == APITypeEnum.RESPONSES:
        client = OpenAIResponses(**config_obj, id=llm_config.model_name)
    else:
        client = OpenAIChat(**config_obj, id=llm_config.model_name)

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.AGNO)
async def litellm_agno(llm_config: LiteLlmModelConfig, _builder: Builder):

    from agno.models.litellm.chat import LiteLLM

    validate_no_responses_api(llm_config, LLMFrameworkEnum.AGNO)

    client = LiteLLM(
        **llm_config.model_dump(
            exclude={"type", "thinking", "model_name", "api_type"},
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        ),
        id=llm_config.model_name,
    )

    yield _patch_llm_based_on_config(client, llm_config)
