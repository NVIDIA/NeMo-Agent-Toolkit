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
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    class CrewAIThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, messages: list[dict[str, str]], *args, **kwargs) -> FunctionArgumentWrapper:
            new_messages = [{"role": "system", "content": self.system_prompt}] + messages
            return FunctionArgumentWrapper(new_messages, *args, **kwargs)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client, CrewAIThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=["call"],
            ))

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    return client


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def azure_openai_crewai(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from crewai import LLM

    # https://docs.crewai.com/en/concepts/llms#azure

    config_obj = {
        **llm_config.model_dump(exclude={
            "type",
            "api_key",
            "azure_endpoint",
            "azure_deployment",
        }, by_alias=True),
    }

    api_key = llm_config.api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
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
    config_obj["model"] = model

    client = LLM(**config_obj)

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def nim_crewai(llm_config: NIMModelConfig, _builder: Builder):

    from crewai import LLM

    config_obj = {
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
        "model": f"nvidia_nim/{llm_config.model_name}",
    }

    # Because CrewAI uses a different environment variable for the API key, we need to set it here manually
    if config_obj.get("api_key") is None and "NVIDIA_NIM_API_KEY" not in os.environ:
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_api_key is not None:
            os.environ["NVIDIA_NIM_API_KEY"] = nvidia_api_key

    client = LLM(**config_obj)

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def openai_crewai(llm_config: OpenAIModelConfig, _builder: Builder):

    from crewai import LLM

    config_obj = {
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
    }

    client = LLM(**config_obj)

    yield _patch_llm_based_on_config(client, llm_config)
