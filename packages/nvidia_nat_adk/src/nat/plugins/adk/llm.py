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

import logging
import os

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.http_client import _handle_litellm_verify_ssl  # ADK uses litellm under the hood
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
            "api_type",
            "azure_deployment",
            "azure_endpoint",
            "max_retries",
            "model",
            "model_name",
            "request_timeout",
            "thinking",
            "type",
            "verify_ssl"
        },
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    if config.azure_endpoint:
        config_dict["api_base"] = config.azure_endpoint
    if config.request_timeout is not None:
        config_dict["timeout"] = config.request_timeout

    config_dict["api_version"] = config.api_version
    _handle_litellm_verify_ssl(config)

    yield LiteLlm(f"azure/{config.azure_deployment}", **config_dict)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def litellm_adk(litellm_config: LiteLlmModelConfig, _builder: Builder):
    from google.adk.models.lite_llm import LiteLlm

    validate_no_responses_api(litellm_config, LLMFrameworkEnum.ADK)

    _handle_litellm_verify_ssl(litellm_config)
    yield LiteLlm(**litellm_config.model_dump(
        exclude={"api_type", "max_retries", "thinking", "type", "verify_ssl"},
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
        exclude={"api_type", "base_url", "max_retries", "model", "model_name", "thinking", "type", "verify_ssl"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    if config.base_url:
        config_dict["api_base"] = config.base_url

    _handle_litellm_verify_ssl(config)

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
        exclude={
            "api_type",
            "base_url",
            "max_retries",
            "model",
            "model_name",
            "request_timeout",
            "thinking",
            "type",
            "verify_ssl"
        },
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )

    if (api_key := config.api_key.get_secret_value() if config.api_key else os.getenv("OPENAI_API_KEY")):
        config_dict["api_key"] = api_key
    if (base_url := config.base_url or os.getenv("OPENAI_BASE_URL")):
        config_dict["api_base"] = base_url
    if config.request_timeout is not None:
        config_dict["timeout"] = config.request_timeout

    _handle_litellm_verify_ssl(config)

    yield LiteLlm(config.model_name, **config_dict)


@register_llm_client(config_type=DynamoModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def dynamo_adk(config: DynamoModelConfig, _builder: Builder):
    """Create and yield a Google ADK LiteLlm client for Dynamo with nvext.agent_hints support.

    When ``enable_nvext_hints`` is True, this client injects Dynamo routing hints via
    nvext.agent_hints in the request body using a custom httpx transport wrapped in an
    AsyncOpenAI client. This gives the same per-request hint injection as the LangChain
    implementation, including dynamic prefix IDs via DynamoPrefixContext.

    Args:
        config (DynamoModelConfig): The configuration for the Dynamo model.
        _builder (Builder): The NAT builder instance.
    """
    import os

    from google.adk.models.lite_llm import LiteLlm
    from openai import AsyncOpenAI

    from nat.llm.dynamo_llm import _create_httpx_client_with_dynamo_hooks

    validate_no_responses_api(config, LLMFrameworkEnum.ADK)

    config_dict = config.model_dump(
        exclude={
            "type",
            "max_retries",
            "thinking",
            "model_name",
            "model",
            "base_url",
            "api_type",
            *DynamoModelConfig.get_dynamo_field_names()
        },
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )

    if config.base_url:
        config_dict["api_base"] = config.base_url

    async with _create_httpx_client_with_dynamo_hooks(config) as http_client:

        api_key = (config.api_key.get_secret_value() if config.api_key else os.getenv("OPENAI_API_KEY", "unused"))
        base_url = config.base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

        openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        config_dict["client"] = openai_client
        yield LiteLlm(config.model_name, **config_dict)
