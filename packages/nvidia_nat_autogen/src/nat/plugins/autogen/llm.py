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
"""AutoGen LLM client registrations for NAT.

This module provides AutoGen-compatible LLM client wrappers for the following providers:

Supported Providers
-------------------
- **OpenAI**: Direct OpenAI API integration via ``OpenAIChatCompletionClient``
- **Azure OpenAI**: Azure-hosted OpenAI models via ``AzureOpenAIChatCompletionClient``
- **NVIDIA NIM**: OpenAI-compatible endpoints for NVIDIA models
- **LiteLLM**: Unified interface to multiple LLM providers via OpenAI-compatible client
- **AWS Bedrock**: Amazon Bedrock models (Claude/Anthropic) via ``AnthropicBedrockChatCompletionClient``

Each wrapper:
- Patches clients with NAT retry logic from ``RetryMixin``
- Injects chain-of-thought prompts when ``ThinkingMixin`` is configured
- Removes NAT-specific config keys before instantiating AutoGen clients
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.common import get_secret_value
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType")


def _patch_autogen_client_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:
    """Patch AutoGen client with NAT mixins (retry, thinking).

    Args:
        client (ModelType): The AutoGen LLM client to patch.
        llm_config (LLMBaseConfig): The LLM configuration containing mixin settings.

    Returns:
        ModelType: The patched AutoGen LLM client.
    """
    from autogen_core.models import SystemMessage

    class AutoGenThinkingInjector(BaseThinkingInjector):
        """Thinking injector for AutoGen message format.

        Injects a system message at the start of the message list to enable
        chain-of-thought prompting for supported models (e.g., Nemotron).
        """

        @override
        def inject(self, messages: list, *args: Any, **kwargs: Any) -> FunctionArgumentWrapper:
            """Inject thinking system prompt into AutoGen messages.

            Args:
                messages (list): List of AutoGen messages (UserMessage, AssistantMessage, SystemMessage)
                *args (Any): Additional positional arguments
                **kwargs (Any): Additional keyword arguments

            Returns:
                FunctionArgumentWrapper: Wrapper containing modified args and kwargs
            """
            system_message = SystemMessage(content=self.system_prompt)
            new_messages = [system_message] + messages
            return FunctionArgumentWrapper(new_messages, *args, **kwargs)

    # Apply retry mixin if configured
    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    # Apply thinking mixin if configured
    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            AutoGenThinkingInjector(system_prompt=llm_config.thinking_system_prompt,
                                    function_names=[
                                        "create",
                                        "create_stream",
                                    ]))

    return client


async def _close_autogen_client(client: Any) -> None:
    """Close an AutoGen client if it has a close method.

    Args:
        client: The AutoGen client to close
    """
    try:
        if hasattr(client, "close"):
            await client.close()
        elif hasattr(client, "_client") and hasattr(client._client, "close"):
            await client._client.close()
    except Exception:
        logger.debug("Error closing AutoGen client", exc_info=True)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def openai_autogen(llm_config: OpenAIModelConfig, _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create OpenAI client for AutoGen integration.

    Args:
        llm_config (OpenAIModelConfig): OpenAI model configuration
        _builder (Builder): NAT builder instance

    Yields:
        AsyncGenerator[ModelType, None]: Configured AutoGen OpenAI client
    """
    from autogen_core.models import ModelFamily
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # Extract AutoGen-compatible configuration
    config_obj = {
        **llm_config.model_dump(
            exclude={"type", "model_name", "thinking"},
            by_alias=True,
            exclude_none=True,
        ),
    }

    # Define model info for AutoGen 0.7.4 (replaces model_capabilities)
    model_info = ModelInfo(vision=False,
                           function_calling=True,
                           json_output=True,
                           family=ModelFamily.UNKNOWN,
                           structured_output=True,
                           multiple_system_messages=True)

    # Add required AutoGen 0.7.4 parameters
    config_obj.update({"model_info": model_info})
    config_obj.pop("model", None)

    # Create AutoGen OpenAI client
    client = OpenAIChatCompletionClient(model=llm_config.model_name, **config_obj)

    try:
        # Apply NAT mixins and yield patched client
        yield _patch_autogen_client_based_on_config(client, llm_config)
    finally:
        await _close_autogen_client(client)


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def azure_openai_autogen(llm_config: AzureOpenAIModelConfig,
                               _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create Azure OpenAI client for AutoGen integration.

    Args:
        llm_config (AzureOpenAIModelConfig): Azure OpenAI model configuration
        _builder (Builder): NAT builder instance

    Yields:
        AsyncGenerator[ModelType, None]: Configured AutoGen Azure OpenAI client
    """
    from autogen_core.models import ModelFamily
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

    config_obj = {
        "api_key":
            llm_config.api_key,
        "base_url":
            f"{llm_config.azure_endpoint}/openai/deployments/{llm_config.azure_deployment}",
        "api_version":
            llm_config.api_version,
        **llm_config.model_dump(
            exclude={"type", "azure_deployment", "thinking", "azure_endpoint", "api_version"},
            by_alias=True,
            exclude_none=True,
        ),
    }

    model_info = ModelInfo(vision=False,
                           function_calling=True,
                           json_output=True,
                           family=ModelFamily.UNKNOWN,
                           structured_output=True,
                           multiple_system_messages=True)

    config_obj.update({"model_info": model_info})

    client = AzureOpenAIChatCompletionClient(
        model=llm_config.azure_deployment,  # Use deployment name for Azure
        **config_obj)

    try:
        # Apply NAT mixins and yield patched client
        yield _patch_autogen_client_based_on_config(client, llm_config)
    finally:
        await _close_autogen_client(client)


def _strip_strict_from_tools_deep(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove 'strict' field from tool definitions in request kwargs for NIM compatibility.

    NIM's API doesn't support OpenAI's 'strict' parameter in tool/function definitions.
    AutoGen adds this field automatically, so we strip it before sending to NIM.

    Args:
        kwargs: The request keyword arguments dictionary

    Returns:
        kwargs with 'strict' field removed from tool function definitions
    """
    tools = kwargs.get("tools")

    # Handle NotGiven sentinel or None - just return unchanged
    if tools is None or not isinstance(tools, list | tuple):
        return kwargs

    kwargs = kwargs.copy()
    cleaned_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_copy = tool.copy()
            if "function" in tool_copy and isinstance(tool_copy["function"], dict):
                func_copy = tool_copy["function"].copy()
                func_copy.pop("strict", None)
                tool_copy["function"] = func_copy
            cleaned_tools.append(tool_copy)
        else:
            cleaned_tools.append(tool)
    kwargs["tools"] = cleaned_tools
    return kwargs


def _patch_nim_client_for_tools(client: ModelType) -> ModelType:
    """Patch AutoGen client's underlying OpenAI client to strip 'strict' from tools for NIM.

    This patches at the lowest level (the actual OpenAI AsyncClient) to ensure
    the 'strict' field is removed after AutoGen's internal processing.

    Args:
        client: The AutoGen OpenAI client to patch

    Returns:
        The patched client (unmodified if patching fails)
    """
    try:
        # Access the underlying OpenAI AsyncClient (protected member)
        openai_client = getattr(client, "_client", None)
        if openai_client is None:
            logger.warning("Unable to patch NIM client for tools - _client attribute not found")
            return client

        # Verify the expected structure exists
        if not hasattr(openai_client, "chat") or not hasattr(openai_client.chat, "completions"):
            logger.warning("Unable to patch NIM client for tools - unexpected client structure")
            return client

        # Patch the chat.completions.create method
        original_create = openai_client.chat.completions.create

        async def patched_create(*args: Any, **kwargs: Any) -> Any:
            # Strip 'strict' from tools before sending to NIM
            kwargs = _strip_strict_from_tools_deep(kwargs)
            return await original_create(*args, **kwargs)

        openai_client.chat.completions.create = patched_create
        return client

    except AttributeError as e:
        logger.warning("Unable to patch NIM client for tools - AutoGen internal structure changed: %s", e)
        return client


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def nim_autogen(llm_config: NIMModelConfig, _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create NVIDIA NIM client for AutoGen integration.

    Args:
        llm_config (NIMModelConfig): NIM model configuration
        _builder (Builder): NAT builder instance

    Yields:
        Configured AutoGen NIM client (via OpenAI compatibility)
    """
    from autogen_core.models import ModelFamily
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # Extract NIM configuration for OpenAI-compatible client
    config_obj = {
        **llm_config.model_dump(
            exclude={"type", "model_name", "thinking"},
            by_alias=True,
            exclude_none=True,
        ),
    }

    # Define model info for AutoGen 0.7.4 (replaces model_capabilities)
    # Note: structured_output=False because NIM doesn't support OpenAI's 'strict' parameter
    model_info = ModelInfo(vision=False,
                           function_calling=True,
                           json_output=True,
                           family=ModelFamily.UNKNOWN,
                           structured_output=False,
                           multiple_system_messages=True)

    # Add required AutoGen 0.7.4 parameters
    config_obj.update({"model_info": model_info})
    config_obj.pop("model", None)

    # NIM uses OpenAI-compatible API
    client = OpenAIChatCompletionClient(model=llm_config.model_name, **config_obj)

    # Patch to remove 'strict' field from tools (NIM doesn't support it)
    client = _patch_nim_client_for_tools(client)

    try:
        # Apply NAT mixins and yield patched client
        yield _patch_autogen_client_based_on_config(client, llm_config)
    finally:
        await _close_autogen_client(client)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def litellm_autogen(llm_config: LiteLlmModelConfig, _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create LiteLLM client for AutoGen integration.

    LiteLLM provides a unified interface to multiple LLM providers. This integration
    uses AutoGen's OpenAI-compatible client since LiteLLM exposes an OpenAI-compatible
    API endpoint.

    Args:
        llm_config (LiteLlmModelConfig): LiteLLM model configuration
        _builder (Builder): NAT builder instance

    Yields:
        AsyncGenerator[ModelType, None]: Configured AutoGen client via LiteLLM
    """
    from autogen_core.models import ModelFamily
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # Extract LiteLLM configuration for OpenAI-compatible client
    config_obj = {
        **llm_config.model_dump(
            exclude={"type", "model_name", "thinking"},
            by_alias=True,
            exclude_none=True,
        ),
    }

    # Resolve API key from secret if provided
    if llm_config.api_key is not None:
        config_obj["api_key"] = get_secret_value(llm_config.api_key)

    # Define model info for AutoGen
    model_info = ModelInfo(vision=False,
                           function_calling=True,
                           json_output=True,
                           family=ModelFamily.UNKNOWN,
                           structured_output=True,
                           multiple_system_messages=True)

    config_obj.update({"model_info": model_info})
    config_obj.pop("model", None)

    # LiteLLM uses OpenAI-compatible API
    client = OpenAIChatCompletionClient(model=llm_config.model_name, **config_obj)

    try:
        # Apply NAT mixins and yield patched client
        yield _patch_autogen_client_based_on_config(client, llm_config)
    finally:
        await _close_autogen_client(client)


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def bedrock_autogen(llm_config: AWSBedrockModelConfig, _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create AWS Bedrock client for AutoGen integration.

    Uses AutoGen's ``AnthropicBedrockChatCompletionClient`` which supports
    Anthropic Claude models hosted on AWS Bedrock. Credentials are loaded in
    the following priority:

    1. Explicit values from ``credentials_profile_name`` in the AWS profile.
    2. Standard environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
       ``AWS_SESSION_TOKEN``).
    3. Ambient credentials provided by the compute environment (IAM role).

    Args:
        llm_config (AWSBedrockModelConfig): AWS Bedrock model configuration
        _builder (Builder): NAT builder instance

    Yields:
        AsyncGenerator[ModelType, None]: Configured AutoGen Bedrock client
    """
    from autogen_ext.models.anthropic import AnthropicBedrockChatCompletionClient

    # Build Bedrock-specific configuration
    bedrock_config: dict[str, Any] = {
        "model": llm_config.model_name,
    }

    # Handle region - None or "None" string should use AWS default
    if llm_config.region_name not in (None, "None"):
        bedrock_config["aws_region"] = llm_config.region_name

    # Add optional parameters if provided
    if llm_config.credentials_profile_name is not None:
        bedrock_config["aws_profile"] = llm_config.credentials_profile_name

    if llm_config.base_url is not None:
        bedrock_config["base_url"] = llm_config.base_url

    # Add model parameters
    if llm_config.max_tokens is not None:
        bedrock_config["max_tokens"] = llm_config.max_tokens

    if llm_config.temperature is not None:
        bedrock_config["temperature"] = llm_config.temperature

    if llm_config.top_p is not None:
        bedrock_config["top_p"] = llm_config.top_p

    # Create AutoGen Bedrock client
    client = AnthropicBedrockChatCompletionClient(**bedrock_config)

    try:
        # Apply NAT mixins and yield patched client
        yield _patch_autogen_client_based_on_config(client, llm_config)
    finally:
        await _close_autogen_client(client)
