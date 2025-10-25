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

import os
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
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
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

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
async def openai_strands(llm_config: OpenAIModelConfig, _builder: Builder):

    from strands.models.openai import OpenAIModel

    params = llm_config.model_dump(exclude={"type", "api_key", "base_url", "model_name"},
                                   by_alias=True,
                                   exclude_none=True)
    # Remove NAT-specific and retry-specific keys not accepted by OpenAI chat.create
    for k in ("max_retries", "num_retries", "retry_on_status_codes", "retry_on_errors", "thinking"):
        params.pop(k, None)

    client = OpenAIModel(
        client_args={
            "api_key": llm_config.api_key or os.getenv("OPENAI_API_KEY"),
            "base_url": llm_config.base_url,
        },
        model_id=llm_config.model_name,
        params=params,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.STRANDS)
async def nim_strands(llm_config: NIMModelConfig, _builder: Builder):

    # NIM is OpenAI compatible; use OpenAI model with NIM base_url and api_key
    from strands.models.openai import OpenAIModel

    # Create a custom OpenAI model that formats text content as strings for NIM compatibility
    class NIMCompatibleOpenAIModel(OpenAIModel):

        @classmethod
        def format_request_messages(cls, messages, system_prompt=None):
            # Get the formatted messages from the parent
            formatted_messages = super().format_request_messages(messages, system_prompt)

            # Convert content arrays with only text to strings for NIM compatibility
            for msg in formatted_messages:
                content = msg.get("content")
                if isinstance(content, list) and len(content) == 1 and isinstance(content[0], str):
                    # If content is a single-item list with a string, flatten it
                    msg["content"] = content[0]
                elif isinstance(content, list) and all(
                        isinstance(item, dict) and item.get("type") == "text" for item in content):
                    # If all items are text blocks, join them into a single string
                    text_content = "".join(item["text"] for item in content)
                    # Ensure we don't send empty strings (NIM rejects them)
                    msg["content"] = text_content if text_content.strip() else " "
                elif isinstance(content, list) and len(content) == 0:
                    # Handle empty content arrays
                    msg["content"] = " "
                elif isinstance(content, str) and not content.strip():
                    # Handle empty strings
                    msg["content"] = " "

            return formatted_messages

    params = llm_config.model_dump(exclude={"type", "api_key", "base_url", "model_name"},
                                   by_alias=True,
                                   exclude_none=True)
    # Remove NAT-specific and retry-specific keys not accepted by OpenAI
    for k in ("max_retries", "num_retries", "retry_on_status_codes", "retry_on_errors", "thinking"):
        params.pop(k, None)

    client = NIMCompatibleOpenAIModel(
        client_args={
            "api_key": llm_config.api_key or os.getenv("NVIDIA_API_KEY"),
            "base_url": llm_config.base_url or "https://integrate.api.nvidia.com/v1",
        },
        model_id=llm_config.model_name,
        params=params,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.STRANDS)
async def bedrock_strands(llm_config: AWSBedrockModelConfig, _builder: Builder):

    from strands.models.bedrock import BedrockModel

    client = BedrockModel(
        model_id=llm_config.model_name,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        region_name=llm_config.region_name,
        endpoint_url=llm_config.base_url,
    )

    yield _patch_llm_based_on_config(client, llm_config)
