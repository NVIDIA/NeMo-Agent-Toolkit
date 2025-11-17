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
import os

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.utils.responses_api import validate_no_responses_api

logger = logging.getLogger(__name__)


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def azure_openai_adk(config: AzureOpenAIModelConfig, _builder: Builder):
    """Create and yield a Google ADK `AzureOpenAI` client from a NAT `AzureOpenAIModelConfig`.

    Args:
        config (AzureOpenAIModelConfig): The configuration for the AzureOpenAI model.
        _builder (Builder): The NAT builder instance.
    """
    from google.adk.models.lite_llm import LiteLlm

    validate_no_responses_api(config, LLMFrameworkEnum.ADK)

    config_dict = config.model_dump(
        exclude={
            "type", "max_retries", "thinking", "azure_endpoint", "azure_deployment", "model_name", "model", "api_type"
        },
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    if config.azure_endpoint:
        config_dict["api_base"] = config.azure_endpoint

    config_dict["api_version"] = config.api_version

    yield LiteLlm(f"azure/{config.azure_deployment}", **config_dict)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def litellm_adk(litellm_config: LiteLlmModelConfig, _builder: Builder):
    from google.adk.models.lite_llm import LiteLlm

    validate_no_responses_api(litellm_config, LLMFrameworkEnum.ADK)

    yield LiteLlm(**litellm_config.model_dump(
        exclude={"type", "max_retries", "thinking", "api_type"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    ))


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def nim_adk(config: NIMModelConfig, _builder: Builder):
    """Create and yield a Google ADK `NIM` client from a NAT `NIMModelConfig`.

    Args:
        config (NIMModelConfig): The configuration for the NIM model.
        _builder (Builder): The NAT builder instance.
    """
    import litellm
    from google.adk.models.lite_llm import LiteLlm

    validate_no_responses_api(config, LLMFrameworkEnum.ADK)

    logger.warning("NIMs do not currently support tools with ADK. Tools will be ignored.")
    litellm.add_function_to_prompt = True
    litellm.drop_params = True

    if (api_key := os.getenv("NVIDIA_API_KEY", None)) is not None:
        os.environ["NVIDIA_NIM_API_KEY"] = api_key

    config_dict = config.model_dump(
        exclude={"type", "max_retries", "thinking", "model_name", "model", "base_url", "api_type"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    if config.base_url:
        config_dict["api_base"] = config.base_url

    yield LiteLlm(f"nvidia_nim/{config.model_name}", **config_dict)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def openai_adk(config: OpenAIModelConfig, _builder: Builder):
    """Create and yield a Google ADK `OpenAI` client from a NAT `OpenAIModelConfig`.

    Args:
        config (OpenAIModelConfig): The configuration for the OpenAI model.
        _builder (Builder): The NAT builder instance.
    """
    from google.adk.models.lite_llm import LiteLlm

    validate_no_responses_api(config, LLMFrameworkEnum.ADK)

    config_dict = config.model_dump(
        exclude={"type", "max_retries", "thinking", "model_name", "model", "base_url", "api_type"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    if config.base_url:
        config_dict["api_base"] = config.base_url

    yield LiteLlm(config.model_name, **config_dict)
