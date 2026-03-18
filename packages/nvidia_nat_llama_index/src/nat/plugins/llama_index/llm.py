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

import os
from collections.abc import Sequence
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.common import get_secret_value
from nat.data_models.llm import APITypeEnum
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.http_client import http_clients
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    from llama_index.core.base.llms.types import ChatMessage

    class LlamaIndexThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, messages: Sequence[ChatMessage], *args, **kwargs) -> FunctionArgumentWrapper:
            for i, message in enumerate(messages):
                if message.role == "system":
                    if self.system_prompt not in str(message.content):
                        messages = list(messages)
                        messages[i] = ChatMessage(role="system", content=f"{message.content}\n{self.system_prompt}")
                    break
            else:
                messages = list(messages)
                messages.insert(0, ChatMessage(role="system", content=self.system_prompt))
            return FunctionArgumentWrapper(messages, *args, **kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            LlamaIndexThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=[
                    "chat",
                    "stream_chat",
                    "achat",
                    "astream_chat",
                ],
            ))

    return client


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def aws_bedrock_llama_index(llm_config: AWSBedrockModelConfig, _builder: Builder):

    from llama_index.llms.bedrock import Bedrock

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    # LlamaIndex uses context_size instead of max_tokens
    llm = Bedrock(**llm_config.model_dump(exclude={"api_type", "thinking", "top_p", "type", "verify_ssl"},
                                          by_alias=True,
                                          exclude_none=True,
                                          exclude_unset=True))

    yield _patch_llm_based_on_config(llm, llm_config)


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def azure_openai_llama_index(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from llama_index.llms.azure_openai import AzureOpenAI

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    config_dict = llm_config.model_dump(
        exclude={"api_type", "api_version", "request_timeout", "thinking", "type", "verify_ssl"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True)
    if llm_config.request_timeout is not None:
        config_dict["timeout"] = llm_config.request_timeout

    async with http_clients(llm_config) as http_clients_dict:
        config_dict.update(http_clients_dict)
        llm = AzureOpenAI(
            **config_dict,
            api_version=llm_config.api_version,
        )

        yield _patch_llm_based_on_config(llm, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def nim_llama_index(llm_config: NIMModelConfig, _builder: Builder):

    from llama_index.llms.nvidia import NVIDIA

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    config_dict = llm_config.model_dump(
        exclude={
            "api_type",
            "thinking",
            "type",
            "verify_ssl",
        },
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )

    async with http_clients(llm_config) as http_clients_dict:
        config_dict.update(http_clients_dict)
        llm = NVIDIA(**config_dict)

        yield _patch_llm_based_on_config(llm, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def openai_llama_index(llm_config: OpenAIModelConfig, _builder: Builder):

    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai import OpenAIResponses

    config_dict = llm_config.model_dump(
        exclude={"api_key", "api_type", "base_url", "request_timeout", "thinking", "type", "verify_ssl"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )

    if (api_key := get_secret_value(llm_config.api_key) or os.getenv("OPENAI_API_KEY")):
        config_dict["api_key"] = api_key
    if (base_url := llm_config.base_url or os.getenv("OPENAI_BASE_URL")):
        # LlamaIndex's OpenAI wrapper expects "api_base" instead of "base_url"
        config_dict["api_base"] = base_url
    if llm_config.request_timeout is not None:
        config_dict["timeout"] = llm_config.request_timeout

    async with http_clients(llm_config) as http_clients_dict:
        config_dict.update(http_clients_dict)
        if llm_config.api_type == APITypeEnum.RESPONSES:
            llm = OpenAIResponses(**config_dict)
        else:
            llm = OpenAI(**config_dict)

        yield _patch_llm_based_on_config(llm, llm_config)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def litellm_llama_index(llm_config: LiteLlmModelConfig, _builder: Builder):

    from llama_index.llms.litellm import LiteLLM

    from nat.llm.utils.http_client import _handle_litellm_verify_ssl

    _handle_litellm_verify_ssl(llm_config)
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)

    llm = LiteLLM(
        **llm_config.model_dump(exclude={"api_type", "thinking", "type", "verify_ssl"},
                                by_alias=True,
                                exclude_none=True,
                                exclude_unset=True), )

    yield _patch_llm_based_on_config(llm, llm_config)
