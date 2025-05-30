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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.llm.aws_bedrock_llm import AWSBedrockModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nim_langchain(llm_config: NIMModelConfig, builder: Builder):

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    yield ChatNVIDIA(**llm_config.model_dump(exclude={"type"}, by_alias=True))


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(llm_config: OpenAIModelConfig, builder: Builder):

    from langchain_openai import ChatOpenAI

    yield ChatOpenAI(**llm_config.model_dump(exclude={"type"}, by_alias=True))


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def aws_bedrock_langchain(llm_config: AWSBedrockModelConfig, builder: Builder):

    from langchain_aws import ChatBedrockConverse

    yield ChatBedrockConverse(**llm_config.model_dump(exclude={"type", "context_size"}, by_alias=True))
