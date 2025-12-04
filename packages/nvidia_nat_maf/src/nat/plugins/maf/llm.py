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

from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.common import get_secret_value
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api
from nat.utils.type_utils import override

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    #TODO: Needs to be implemented as per MAF

    return client


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def azure_openai_semantic_kernel(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    validate_no_responses_api(llm_config, LLMFrameworkEnum.SEMANTIC_KERNEL)


    from agent_framework.azure import AzureOpenAIChatClient
    # The AzureKeyCredential is required for API key authentication
    from azure.core.credentials import AzureKeyCredential

    credential = AzureKeyCredential(get_secret_value(llm_config.api_key))

    llm = AzureOpenAIChatClient(
                # Use 'endpoint' for the API base/URL
                endpoint=llm_config.azure_endpoint,
                # Pass the credential object
                #credential=credential,
                api_key=get_secret_value(llm_config.api_key),
                # Specify the deployment name
                deployment_name=llm_config.azure_deployment,
                # For this client, the model_name argument is less critical than deployment_name
                model_name=llm_config.azure_deployment
            )

    yield _patch_llm_based_on_config(llm, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def openai_semantic_kernel(llm_config: OpenAIModelConfig, _builder: Builder):

    from agent_framework.openai import OpenAIResponsesClient

    validate_no_responses_api(llm_config, LLMFrameworkEnum.SEMANTIC_KERNEL)

    llm = OpenAIResponsesClient()

    yield _patch_llm_based_on_config(llm, llm_config)
