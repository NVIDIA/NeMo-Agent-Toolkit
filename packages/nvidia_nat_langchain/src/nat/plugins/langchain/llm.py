# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Sequence
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.prompt_values import PromptValue

    class LangchainThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, messages: LanguageModelInput, *args, **kwargs) -> FunctionArgumentWrapper:
            """
            Inject a system prompt into the messages.

            The messages are the first (non-object) argument to the function.
            The rest of the arguments are passed through unchanged.

            Args:
                messages: The messages to inject the system prompt into.
                *args: The rest of the arguments to the function.
                **kwargs: The rest of the keyword arguments to the function.

            Returns:
                FunctionArgumentWrapper: An object that contains the transformed args and kwargs.

            Raises:
                ValueError: If the messages are not a valid type for LanguageModelInput.
            """
            system_message = SystemMessage(content=self.system_prompt)
            if isinstance(messages, BaseMessage):
                new_messages = [system_message, messages]
                return FunctionArgumentWrapper(new_messages, *args, **kwargs)
            elif isinstance(messages, PromptValue):
                new_messages = [system_message, *messages.to_messages()]
                return FunctionArgumentWrapper(new_messages, *args, **kwargs)
            elif isinstance(messages, str):
                new_messages = [system_message, HumanMessage(content=messages)]
                return FunctionArgumentWrapper(new_messages, *args, **kwargs)
            elif isinstance(messages, Sequence):
                if all(isinstance(m, BaseMessage) for m in messages):
                    new_messages = [system_message, *list(messages)]
                    return FunctionArgumentWrapper(new_messages, *args, **kwargs)
                raise ValueError(
                    "Unsupported sequence element types for LanguageModelInput; expected Sequence[BaseMessage].")
            else:
                return FunctionArgumentWrapper(messages, *args, **kwargs)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            LangchainThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=[
                    "invoke",
                    "ainvoke",
                    "stream",
                    "astream",
                ],
            ))

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    return client


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def aws_bedrock_langchain(llm_config: AWSBedrockModelConfig, _builder: Builder):

    from langchain_aws import ChatBedrockConverse

    client = ChatBedrockConverse(**llm_config.model_dump(exclude={"type", "context_size"}, by_alias=True))

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def azure_openai_langchain(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from langchain_openai import AzureChatOpenAI

    client = AzureChatOpenAI(**llm_config.model_dump(exclude={"type"}, by_alias=True))

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nim_langchain(llm_config: NIMModelConfig, _builder: Builder):

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # prefer max_completion_tokens over max_tokens
    client = ChatNVIDIA(
        **llm_config.model_dump(exclude={"type", "max_tokens"}, by_alias=True),
        max_completion_tokens=llm_config.max_tokens,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(llm_config: OpenAIModelConfig, _builder: Builder):

    from langchain_openai import ChatOpenAI

    # If stream_usage is specified, it will override the default value of True.
    client = ChatOpenAI(stream_usage=True, **llm_config.model_dump(exclude={"type"}, by_alias=True))

    yield _patch_llm_based_on_config(client, llm_config)
