# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_embedder_client
from nat.data_models.retry_mixin import RetryMixin
from nat.embedder.azure_openai_embedder import AzureOpenAIEmbedderModelConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.llm.utils.http_client import http_clients
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


@register_embedder_client(config_type=AzureOpenAIEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def azure_openai_llama_index(embedder_config: AzureOpenAIEmbedderModelConfig, _builder: Builder):

    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

    async with http_clients(embedder_config) as http_clients_dict:
        client = AzureOpenAIEmbedding(
            **embedder_config.model_dump(exclude={"api_version", "type", "verify_ssl"},
                                         by_alias=True,
                                         exclude_none=True,
                                         exclude_unset=True),
            api_version=embedder_config.api_version,
            **http_clients_dict,
        )

        if isinstance(embedder_config, RetryMixin):
            client = patch_with_retry(client,
                                      retries=embedder_config.num_retries,
                                      retry_codes=embedder_config.retry_on_status_codes,
                                      retry_on_messages=embedder_config.retry_on_errors)

        yield client


@register_embedder_client(config_type=NIMEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def nim_llama_index(embedder_config: NIMEmbedderModelConfig, _builder: Builder):

    from llama_index.embeddings.nvidia import NVIDIAEmbedding  # pylint: disable=no-name-in-module

    if not embedder_config.verify_ssl:
        # Remove once https://github.com/run-llama/llama_index/issues/20945 is resolved
        raise ValueError("verify_ssl is currently not supported for NVIDIAEmbedding.")

    client = NVIDIAEmbedding(
        **embedder_config.model_dump(exclude={"model_name", "type", "verify_ssl"},
                                     by_alias=True,
                                     exclude_none=True,
                                     exclude_unset=True),
        model=embedder_config.model_name,
    )

    if isinstance(embedder_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=embedder_config.num_retries,
                                  retry_codes=embedder_config.retry_on_status_codes,
                                  retry_on_messages=embedder_config.retry_on_errors)

    yield client


@register_embedder_client(config_type=OpenAIEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def openai_llama_index(embedder_config: OpenAIEmbedderModelConfig, _builder: Builder):

    from llama_index.embeddings.openai import OpenAIEmbedding

    async with http_clients(embedder_config) as http_clients_dict:
        client = OpenAIEmbedding(
            **embedder_config.model_dump(exclude={"type", "verify_ssl"},
                                         by_alias=True,
                                         exclude_none=True,
                                         exclude_unset=True),
            **http_clients_dict,
        )

        if isinstance(embedder_config, RetryMixin):
            client = patch_with_retry(client,
                                      retries=embedder_config.num_retries,
                                      retry_codes=embedder_config.retry_on_status_codes,
                                      retry_on_messages=embedder_config.retry_on_errors)

        yield client
