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
from nat.data_models.common import get_secret_value
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    from semantic_kernel.contents.chat_history import ChatHistory

    class SemanticKernelThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, chat_history: ChatHistory, *args, **kwargs) -> FunctionArgumentWrapper:
            """
            Inject a system prompt into the chat_history.

            The chat_history is the first (non-object) argument to the function.
            The rest of the arguments are passed through unchanged.

            Args:
                chat_history: The ChatHistory object to inject the system prompt into.
                *args: The rest of the arguments to the function.
                **kwargs: The rest of the keyword arguments to the function.

            Returns:
                FunctionArgumentWrapper: An object that contains the transformed args and kwargs.
            """
            if chat_history.system_message is None:
                new_messages = ChatHistory(chat_history.messages, system_message=self.system_prompt)
                return FunctionArgumentWrapper(new_messages, *args, **kwargs)
            else:
                new_messages = ChatHistory(
                    chat_history.messages,
                    system_message=f"{self.system_prompt}\n\n{chat_history.system_message}",
                )
                return FunctionArgumentWrapper(new_messages, *args, **kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            SemanticKernelThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=[
                    "get_chat_message_contents",
                    "get_streaming_chat_message_contents",
                ],
            ))

    return client


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def azure_openai_semantic_kernel(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    validate_no_responses_api(llm_config, LLMFrameworkEnum.SEMANTIC_KERNEL)

    llm = AzureChatCompletion(
        api_key=get_secret_value(llm_config.api_key),
        api_version=llm_config.api_version,
        endpoint=llm_config.azure_endpoint,
        deployment_name=llm_config.azure_deployment,
    )

    yield _patch_llm_based_on_config(llm, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def openai_semantic_kernel(llm_config: OpenAIModelConfig, _builder: Builder):

    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    validate_no_responses_api(llm_config, LLMFrameworkEnum.SEMANTIC_KERNEL)

    llm = OpenAIChatCompletion(ai_model_id=llm_config.model_name)

    yield _patch_llm_based_on_config(llm, llm_config)
