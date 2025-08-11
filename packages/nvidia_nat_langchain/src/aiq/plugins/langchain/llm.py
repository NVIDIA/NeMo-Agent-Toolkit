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
# pylint: disable=unused-argument

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.data_models.llm import APITypeEnum
from aiq.data_models.retry_mixin import RetryMixin
from aiq.llm.aws_bedrock_llm import AWSBedrockModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig
from aiq.utils.exception_handlers.automatic_retries import patch_with_retry

logger = logging.getLogger(__name__)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nim_langchain(llm_config: NIMModelConfig, builder: Builder):

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    if llm_config.api_type != APITypeEnum.CHAT_COMPLETION:
        raise ValueError("NVIDIA AI Endpoints only supports chat completion API type. "
                         f"Received: {llm_config.api_type}")

    client = ChatNVIDIA(**llm_config.model_dump(exclude={"type"}, by_alias=True))

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    yield client


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(llm_config: OpenAIModelConfig, builder: Builder):

    from langchain_openai import ChatOpenAI

    # Default kwargs for OpenAI to include usage metadata in the response. If the user has set stream_usage to False, we
    # will not include this.
    default_kwargs = {"stream_usage": True}
    exclude = {"type"}
    if llm_config.model_name.startswith('o'):
        exclude.add("temperature")

    kwargs = {**default_kwargs, **llm_config.model_dump(exclude=exclude, by_alias=True)}

    if llm_config.api_type == APITypeEnum.RESPONSES:
        kwargs["use_responses_api"] = True
        kwargs["use_previous_response_id"] = True
        if "stream" in kwargs and kwargs["stream"]:
            kwargs["stream"] = False
            logger.warning("Streaming is not supported with the OpenAI Responses API. "
                           "Setting stream to False.")

    client = ChatOpenAI(**kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    yield client


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def aws_bedrock_langchain(llm_config: AWSBedrockModelConfig, builder: Builder):

    from langchain_aws import ChatBedrockConverse

    if llm_config.api_type != APITypeEnum.CHAT_COMPLETION:
        raise ValueError("AWS Bedrock only supports chat completion API type. "
                         f"Received: {llm_config.api_type}")

    client = ChatBedrockConverse(**llm_config.model_dump(exclude={"type", "context_size"}, by_alias=True))

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    yield client
