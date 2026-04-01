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

"""LLM client wrappers for AG2.

AG2 uses LLMConfig (a configuration object) rather than a separate
client class. Each provider wrapper yields an LLMConfig configured
for that provider.
"""

from collections.abc import AsyncGenerator

from autogen import LLMConfig

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.common import get_secret_value
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig


@register_llm_client(
    config_type=OpenAIModelConfig,
    wrapper_type=LLMFrameworkEnum.AG2,
)
async def openai_ag2(
    llm_config: OpenAIModelConfig,
    _builder: Builder,
) -> AsyncGenerator[LLMConfig, None]:
    """Yield an AG2 LLMConfig for OpenAI."""
    config_dict = {
        "api_type": "openai",
        "model": llm_config.model_name,
        "api_key": get_secret_value(llm_config.api_key),
    }
    if llm_config.base_url:
        config_dict["base_url"] = llm_config.base_url

    yield LLMConfig(
        config_dict,
        temperature=llm_config.temperature,
    )


@register_llm_client(
    config_type=NIMModelConfig,
    wrapper_type=LLMFrameworkEnum.AG2,
)
async def nim_ag2(
    llm_config: NIMModelConfig,
    _builder: Builder,
) -> AsyncGenerator[LLMConfig, None]:
    """Yield an AG2 LLMConfig for NVIDIA NIM."""
    config_dict = {
        "api_type": "openai",
        "model": llm_config.model_name,
        "api_key": get_secret_value(llm_config.api_key),
        "base_url": (
            llm_config.base_url
            or "https://integrate.api.nvidia.com/v1"
        ),
    }

    yield LLMConfig(
        config_dict,
        temperature=llm_config.temperature,
    )


@register_llm_client(
    config_type=AzureOpenAIModelConfig,
    wrapper_type=LLMFrameworkEnum.AG2,
)
async def azure_ag2(
    llm_config: AzureOpenAIModelConfig,
    _builder: Builder,
) -> AsyncGenerator[LLMConfig, None]:
    """Yield an AG2 LLMConfig for Azure OpenAI."""
    if not llm_config.azure_endpoint:
        raise ValueError(
            "azure_endpoint is required for Azure OpenAI configs"
        )
    config_dict = {
        "api_type": "azure",
        "model": llm_config.azure_deployment,
        "api_key": get_secret_value(llm_config.api_key),
        "base_url": llm_config.azure_endpoint,
        "api_version": llm_config.api_version,
    }

    yield LLMConfig(
        config_dict,
        temperature=llm_config.temperature,
    )
