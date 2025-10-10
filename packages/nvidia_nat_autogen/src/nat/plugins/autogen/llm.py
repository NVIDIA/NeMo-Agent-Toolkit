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
"""AutoGen LLM client registrations for NAT."""

from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import (BaseThinkingInjector, FunctionArgumentWrapper, patch_with_thinking)
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.type_utils import override

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
        Injects a system message at the start of the message list.

        Args:
            system_prompt: The system prompt to inject.
            *args: The rest of the arguments to the function.
            **kwargs: The rest of the keyword arguments to the function.

        Returns:
            FunctionArgumentWrapper: An object that contains the transformed args and kwargs.

        """

        @override
        def inject(self, messages: list, *args: Any, **kwargs: Any) -> FunctionArgumentWrapper:
            """Inject thinking system prompt into AutoGen messages.
            Args:
                messages (list): List of AutoGen messages (UserMessage, AssistantMessage, SystemMessage
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
                                        "acreate",
                                        "create_stream",
                                        "acreate_stream",
                                    ]))

    return client


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def openai_autogen(llm_config: OpenAIModelConfig, _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create OpenAI client for AutoGen integration.

    Args:
        llm_config (OpenAIModelConfig): OpenAI model configuration
        _builder (Builder): NAT builder instance

    Yields:
        AsyncGenerator[ModelType, None]: Configured AutoGen OpenAI client
    """
    from autogen_core.models import ModelFamily, ModelInfo
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

    # Apply NAT mixins and yield patched client
    yield _patch_autogen_client_based_on_config(client, llm_config)


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
    from autogen_core.models import ModelFamily, ModelInfo
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

    # Apply NAT mixins and yield patched client
    yield _patch_autogen_client_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.AUTOGEN)
async def nim_autogen(llm_config: NIMModelConfig, _builder: Builder) -> AsyncGenerator[ModelType, None]:
    """Create NVIDIA NIM client for AutoGen integration.

    Args:
        llm_config (NIMModelConfig): NIM model configuration
        _builder (Builder): NAT builder instance

    Yields:
        Configured AutoGen NIM client (via OpenAI compatibility)
    """
    from autogen_core.models import ModelFamily, ModelInfo
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
    model_info = ModelInfo(vision=False,
                           function_calling=True,
                           json_output=True,
                           family=ModelFamily.UNKNOWN,
                           structured_output=True,
                           multiple_system_messages=True)

    # Add required AutoGen 0.7.4 parameters
    config_obj.update({"model_info": model_info})

    # NIM uses OpenAI-compatible API
    client = OpenAIChatCompletionClient(model=llm_config.model_name, **config_obj)

    # Apply NAT mixins and yield patched client
    yield _patch_autogen_client_based_on_config(client, llm_config)
