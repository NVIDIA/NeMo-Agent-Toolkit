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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.llm.aws_bedrock_llm import AWSBedrockModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def nim_llama_index(llm_config: NIMModelConfig, builder: Builder):

    from llama_index.llms.nvidia import NVIDIA

    kwargs = llm_config.model_dump(exclude={"type"}, by_alias=True)

    if ("base_url" in kwargs and kwargs["base_url"] is None):
        del kwargs["base_url"]

    llm = NVIDIA(**kwargs)

    yield llm


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def openai_llama_index(llm_config: OpenAIModelConfig, builder: Builder):

    from llama_index.llms.openai import OpenAI

    kwargs = llm_config.model_dump(exclude={"type"}, by_alias=True)

    if ("base_url" in kwargs and kwargs["base_url"] is None):
        del kwargs["base_url"]

    llm = OpenAI(**kwargs)

    yield llm


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def aws_bedrock_llama_index(llm_config: AWSBedrockModelConfig, builder: Builder):

    from llama_index.llms.bedrock import Bedrock

    kwargs = llm_config.model_dump(exclude={"type", "max_tokens"}, by_alias=True)

    llm = Bedrock(**kwargs)

    yield llm
