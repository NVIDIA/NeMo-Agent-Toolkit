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

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.data_models.llm import APITypeEnum
from aiq.llm.aws_bedrock_llm import AWSBedrockModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig

logger = logging.getLogger(__name__)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def nim_llama_index(llm_config: NIMModelConfig, builder: Builder):

    from llama_index.llms.nvidia import NVIDIA

    if llm_config.api_type != APITypeEnum.CHAT_COMPLETION:
        raise ValueError("NVIDIA AI Endpoints only supports chat completion API type. "
                         f"Received: {llm_config.api_type}")

    kwargs = llm_config.model_dump(exclude={"type", "api_type"}, by_alias=True)

    if ("base_url" in kwargs and kwargs["base_url"] is None):
        del kwargs["base_url"]

    llm = NVIDIA(**kwargs)

    yield llm


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def openai_llama_index(llm_config: OpenAIModelConfig, builder: Builder):

    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai import OpenAIResponses

    kwargs = llm_config.model_dump(exclude={"type", "api_type"}, by_alias=True)

    if ("base_url" in kwargs and kwargs["base_url"] is None):
        del kwargs["base_url"]

    if llm_config.api_type == APITypeEnum.CHAT_COMPLETION:

        llm = OpenAI(**kwargs)
        yield llm

    elif llm_config.api_type == APITypeEnum.RESPONSES:
        logger.warning("LLama Index OpenAIResponses class does not support aiq callback handlers. "
                       "Intermediate steps will not be logged. ")

        llm = OpenAIResponses(**kwargs)
        yield llm

    raise ValueError(f"Unsupported API type for OpenAI LLM: {llm_config.api_type}. "
                     "Supported types are: "
                     f"{APITypeEnum.CHAT_COMPLETION}, {APITypeEnum.RESPONSES}.")


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def aws_bedrock_llama_index(llm_config: AWSBedrockModelConfig, builder: Builder):

    from llama_index.llms.bedrock import Bedrock

    if llm_config.api_type != APITypeEnum.CHAT_COMPLETION:
        raise ValueError("AWS Bedrock only supports chat completion API type. "
                         f"Received: {llm_config.api_type}")

    kwargs = llm_config.model_dump(exclude={"type", "max_tokens", "api_type"}, by_alias=True)

    llm = Bedrock(**kwargs)

    yield llm
