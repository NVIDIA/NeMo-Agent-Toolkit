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
from nat.data_models.common import get_secret_value
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
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

    class CrewAIThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, messages: list[dict[str, str]], *args, **kwargs) -> FunctionArgumentWrapper:
            # Attempt to inject the system prompt into the first system message
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    if self.system_prompt not in message["content"]:
                        messages = list(messages)
                        messages[i] = {"role": "system", "content": f"{message['content']}\n{self.system_prompt}"}
                    break
            else:
                messages = list(messages)
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            return FunctionArgumentWrapper(messages, *args, **kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client, CrewAIThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=["call"],
            ))

    return client


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def azure_openai_crewai(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from crewai import LLM

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    # https://docs.crewai.com/en/concepts/llms#azure

    api_key = get_secret_value(llm_config.api_key) if llm_config.api_key else os.environ.get(
        "AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
    if api_key is None:
        raise ValueError("Azure API key is not set")
    os.environ["AZURE_API_KEY"] = api_key
    api_base = (llm_config.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
                or os.environ.get("AZURE_API_BASE"))
    if api_base is None:
        raise ValueError("Azure endpoint is not set")
    os.environ["AZURE_API_BASE"] = api_base

    os.environ["AZURE_API_VERSION"] = llm_config.api_version
    model = llm_config.azure_deployment or os.environ.get("AZURE_MODEL_DEPLOYMENT")
    if model is None:
        raise ValueError("Azure model deployment is not set")

    client = LLM(
        **llm_config.model_dump(
            exclude={"type", "api_key", "azure_endpoint", "azure_deployment", "thinking", "api_type", "api_version"},
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        ),
        model=model,
        api_version=llm_config.api_version,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def nim_crewai(llm_config: NIMModelConfig, _builder: Builder):

    from crewai import LLM

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    # Because CrewAI uses a different environment variable for the API key, we need to set it here manually
    if llm_config.api_key is None and "NVIDIA_NIM_API_KEY" not in os.environ:
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_api_key is not None:
            os.environ["NVIDIA_NIM_API_KEY"] = nvidia_api_key

    client = LLM(
        **llm_config.model_dump(
            exclude={"type", "model_name", "thinking", "api_type"},
            by_alias=True,
            exclude_none=True,
            exclude_unset=True,
        ),
        model=f"nvidia_nim/{llm_config.model_name}",
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def openai_crewai(llm_config: OpenAIModelConfig, _builder: Builder):

    from crewai import LLM

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    client = LLM(**llm_config.model_dump(
        exclude={"type", "thinking", "api_type"}, by_alias=True, exclude_none=True, exclude_unset=True))

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def litellm_crewai(llm_config: LiteLlmModelConfig, _builder: Builder):

    from crewai import LLM

    validate_no_responses_api(llm_config, LLMFrameworkEnum.CREWAI)

    client = LLM(**llm_config.model_dump(
        exclude={"type", "thinking", "api_type"}, by_alias=True, exclude_none=True, exclude_unset=True))

    yield _patch_llm_based_on_config(client, llm_config)
