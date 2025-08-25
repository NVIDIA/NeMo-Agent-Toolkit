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
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

ModelType = TypeVar("ModelType")


def semantic_kernel_thinking_injector(client: ModelType, system_prompt: str) -> ModelType:
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole

    def injector(messages: ChatHistory) -> ChatHistory:
        if messages.system_message is None:
            return ChatHistory(
                messages=messages.messages,
                system_message=system_prompt,
            )
        else:
            return ChatHistory(
                messages=[ChatMessageContent(role=AuthorRole.SYSTEM, content=system_prompt)] + messages.messages,
                system_message=messages.system_message,
            )

    return patch_with_thinking(
        client,
        function_names=["get_chat_message_contents", "get_streaming_chat_message_contents"],
        system_prompt_injector=injector,
    )


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def azure_openai_semantic_kernel(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    llm = AzureChatCompletion(
        api_key=llm_config.api_key,
        api_version=llm_config.api_version,
        endpoint=llm_config.azure_endpoint,
        deployment_name=llm_config.azure_deployment,
    )

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        llm = semantic_kernel_thinking_injector(llm, llm_config.thinking_system_prompt)

    if isinstance(llm_config, RetryMixin):
        llm = patch_with_retry(llm,
                               retries=llm_config.num_retries,
                               retry_codes=llm_config.retry_on_status_codes,
                               retry_on_messages=llm_config.retry_on_errors)

    yield llm


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def openai_semantic_kernel(llm_config: OpenAIModelConfig, _builder: Builder):

    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    llm = OpenAIChatCompletion(ai_model_id=llm_config.model_name)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        llm = semantic_kernel_thinking_injector(llm, llm_config.thinking_system_prompt)

    if isinstance(llm_config, RetryMixin):
        llm = patch_with_retry(llm,
                               retries=llm_config.num_retries,
                               retry_codes=llm_config.retry_on_status_codes,
                               retry_on_messages=llm_config.retry_on_errors)

    yield llm
