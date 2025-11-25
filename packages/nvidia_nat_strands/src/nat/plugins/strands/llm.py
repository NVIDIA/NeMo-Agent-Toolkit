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
"""LLM provider wrappers for AWS Strands integration with NVIDIA NeMo Agent toolkit.

This module provides Strands-compatible LLM client wrappers for the following providers:

Supported Providers
-------------------
- **OpenAI**: Direct OpenAI API integration through ``OpenAIModelConfig``
- **NVIDIA NIM**: OpenAI-compatible endpoints for NVIDIA models through ``NIMModelConfig``
- **AWS Bedrock**: Amazon Bedrock models (such as Claude) through ``AWSBedrockModelConfig``

Each wrapper:
- Validates that Responses API features are disabled (Strands manages tool execution)
- Patches clients with NeMo Agent toolkit retry logic from ``RetryMixin``
- Injects chain-of-thought prompts when ``ThinkingMixin`` is configured
- Removes NeMo Agent toolkit-specific config keys before instantiating Strands clients

Future Provider Support
-----------------------
The following providers are not yet supported but could be contributed:

- **Azure OpenAI**: Would require a Strands Azure OpenAI client wrapper similar to the
  existing OpenAI integration. Contributors should follow the pattern established in
  ``openai_strands`` and ensure Azure-specific authentication (endpoint, API version,
  deployment name) is properly handled.

- **LiteLLM**: The wrapper  would need to handle LiteLLM's unified interface across
  multiple providers while preserving Strands' tool execution semantics.

See the Strands documentation at https://strandsagents.com for model provider details.
"""

import os
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
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:
    """Patch a Strands client per NAT config (retries/thinking) and return it.

    Args:
        client: Concrete Strands model client instance.
        llm_config: NAT LLM config with Retry/Thinking mixins.

    Returns:
        The patched client instance.
    """

    class StrandsThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, messages, *args, **kwargs) -> FunctionArgumentWrapper:
            thinking_prompt = self.system_prompt
            if not thinking_prompt:
                return FunctionArgumentWrapper(messages, *args, **kwargs)

            # Strands calls: model.stream(messages, tool_specs, system_prompt)
            # So system_prompt is the 3rd positional argument (index 1 in *args)
            new_args = list(args)
            new_kwargs = dict(kwargs)

            # Check if system_prompt is passed as positional argument
            if len(new_args) >= 2:  # tool_specs, system_prompt
                existing_system_prompt = new_args[1] or ""  # system_prompt
                if existing_system_prompt:
                    # Prepend thinking prompt to existing system prompt
                    combined_prompt = f"{thinking_prompt}\n\n{existing_system_prompt}"
                else:
                    combined_prompt = thinking_prompt
                new_args[1] = combined_prompt
            elif "system_prompt" in new_kwargs:
                # system_prompt passed as keyword argument
                existing_system_prompt = new_kwargs["system_prompt"] or ""
                if existing_system_prompt:
                    combined_prompt = f"{thinking_prompt}\n\n{existing_system_prompt}"
                else:
                    combined_prompt = thinking_prompt
                new_kwargs["system_prompt"] = combined_prompt
            else:
                # No system_prompt provided, add as keyword argument
                new_kwargs["system_prompt"] = thinking_prompt

            return FunctionArgumentWrapper(messages, *new_args, **new_kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            StrandsThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=[
                    "stream",
                    "structured_output",
                ],
            ))

    return client


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.STRANDS)
async def openai_strands(llm_config: OpenAIModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build a Strands OpenAI client from an NVIDIA NeMo Agent toolkit configuration.

    The wrapper requires the ``nvidia-nat[strands]`` extra and a valid OpenAI-compatible
    API key. When ``llm_config.api_key`` is empty, the integration falls back to the
    ``OPENAI_API_KEY`` environment variable. Responses API features are disabled through
    ``validate_no_responses_api`` because Strands handles tool execution inside the
    framework runtime. The yielded client is patched with NeMo Agent toolkit retry and
    thinking hooks so that framework-level policies remain consistent.

    Args:
        llm_config: OpenAI configuration declared in the workflow.
        _builder: Builder instance provided by the workflow factory (unused).

    Yields:
        Strands ``OpenAIModel`` objects ready to stream responses with NeMo Agent toolkit
        retry/thinking behaviors applied.
    """

    validate_no_responses_api(llm_config, LLMFrameworkEnum.STRANDS)

    from strands.models.openai import OpenAIModel

    params = llm_config.model_dump(
        exclude={"type", "api_type", "api_key", "base_url", "model_name", "max_retries", "thinking"},
        by_alias=True,
        exclude_none=True)

    client = OpenAIModel(
        client_args={
            "api_key": get_secret_value(llm_config.api_key) or os.getenv("OPENAI_API_KEY"),
            "base_url": llm_config.base_url,
        },
        model_id=llm_config.model_name,
        params=params,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.STRANDS)
async def nim_strands(llm_config: NIMModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build a Strands OpenAI-compatible client for NVIDIA NIM endpoints.

    Install the ``nvidia-nat[strands]`` extra and provide a NIM API key either through
    ``llm_config.api_key`` or the ``NVIDIA_API_KEY`` environment variable. The wrapper
    uses the OpenAI-compatible Strands client so Strands can route tool calls while the
    NeMo Agent toolkit continues to manage retries, timeouts, and optional thinking
    prompts. Responses API options are blocked to avoid conflicting execution models.

    Args:
        llm_config: Configuration for calling NVIDIA NIM by way of the OpenAI protocol.
        _builder: Builder instance supplied during workflow construction (unused).

    Yields:
        Patched Strands clients that stream responses using the NVIDIA NIM endpoint
        configured in ``llm_config``.
    """

    validate_no_responses_api(llm_config, LLMFrameworkEnum.STRANDS)

    # NIM is OpenAI compatible; use OpenAI model with NIM base_url and api_key
    from strands.models.openai import OpenAIModel

    # Create a custom OpenAI model that formats text content as strings for NIM compatibility
    class NIMCompatibleOpenAIModel(OpenAIModel):

        @classmethod
        def format_request_message_content(cls, content):
            """Format OpenAI compatible content block with reasoning support.

            Args:
                content: Message content.

            Returns:
                OpenAI compatible content block.

            Raises:
                TypeError: If the content block type cannot be converted to
                    an OpenAI-compatible format.
            """
            # Handle reasoning content by extracting the text
            if isinstance(content, dict) and "reasoningContent" in content:
                reasoning_text = content["reasoningContent"].get("reasoningText", {}).get("text", "")
                return {"text": reasoning_text, "type": "text"}

            # Fall back to parent implementation for other content types
            return super().format_request_message_content(content)

        @classmethod
        def format_request_messages(cls, messages, system_prompt=None, *, system_prompt_content=None, **kwargs):
            # Get the formatted messages from the parent
            formatted_messages = super().format_request_messages(messages,
                                                                 system_prompt,
                                                                 system_prompt_content=system_prompt_content,
                                                                 **kwargs)

            # Convert content arrays with only text to strings for NIM
            # compatibility
            for msg in formatted_messages:
                content = msg.get("content")
                if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], str)):
                    # If content is a single-item list with a string, flatten it
                    msg["content"] = content[0]
                elif (isinstance(content, list)
                      and all(isinstance(item, dict) and item.get("type") == "text" for item in content)):
                    # If all items are text blocks, join them into a single
                    # string
                    text_content = "".join(item["text"] for item in content)
                    # Ensure we don't send empty strings (NIM rejects them)
                    msg["content"] = (text_content if text_content.strip() else " ")
                elif isinstance(content, list) and len(content) == 0:
                    # Handle empty content arrays
                    msg["content"] = " "
                elif isinstance(content, str) and not content.strip():
                    # Handle empty strings
                    msg["content"] = " "

            return formatted_messages

    params = llm_config.model_dump(
        exclude={"type", "api_type", "api_key", "base_url", "model_name", "max_retries", "thinking"},
        by_alias=True,
        exclude_none=True)

    # Determine base_url
    base_url = llm_config.base_url or "https://integrate.api.nvidia.com/v1"

    # Determine api_key; use dummy key for custom NIM endpoints without authentication
    # If base_url is populated (not None) and no API key is available, use a dummy value
    api_key = get_secret_value(llm_config.api_key) or os.getenv("NVIDIA_API_KEY")
    if llm_config.base_url and llm_config.base_url.strip() and api_key is None:
        api_key = "dummy-api-key"

    client = NIMCompatibleOpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": base_url,
        },
        model_id=llm_config.model_name,
        params=params,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.STRANDS)
async def bedrock_strands(llm_config: AWSBedrockModelConfig, _builder: Builder) -> AsyncGenerator[Any, None]:
    """Build a Strands Bedrock client from an NVIDIA NeMo Agent toolkit configuration.

    The integration expects the ``nvidia-nat[strands]`` extra plus AWS credentials that
    can be resolved by ``boto3``. Credentials are loaded in the following priority:

    1. Explicit values embedded in the active AWS profile referenced by
       ``llm_config.credentials_profile_name``.
    2. Standard environment variables such as ``AWS_ACCESS_KEY_ID``,
       ``AWS_SECRET_ACCESS_KEY``, and ``AWS_SESSION_TOKEN``.
    3. Ambient credentials provided by the compute environment (for example, an IAM role
       attached to the container or instance).

    When ``llm_config.region_name`` is ``"None"`` or ``None`` Strands uses the regional
    default configured in AWS. Responses API options remain unsupported so that Strands
    can own tool execution. Retry and thinking hooks are added automatically before the
    Bedrock client is yielded.

    Args:
        llm_config: AWS Bedrock configuration saved in the workflow.
        _builder: Builder reference supplied by the workflow factory (unused).

    Yields:
        Strands ``BedrockModel`` instances configured for the selected Bedrock
        ``model_name`` and patched with NeMo Agent toolkit retry/thinking helpers.
    """

    validate_no_responses_api(llm_config, LLMFrameworkEnum.STRANDS)

    from strands.models.bedrock import BedrockModel

    params = llm_config.model_dump(
        exclude={
            "type",
            "api_type",
            "model_name",
            "region_name",
            "base_url",
            "max_retries",
            "thinking",
            "context_size",
            "credentials_profile_name",
        },
        by_alias=True,
        exclude_none=True,
    )

    region = None if llm_config.region_name in (None, "None") else llm_config.region_name
    client = BedrockModel(model_id=llm_config.model_name,
                          region_name=region,
                          endpoint_url=llm_config.base_url,
                          **params)

    yield _patch_llm_based_on_config(client, llm_config)
