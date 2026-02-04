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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_embedder_client
from nat.data_models.retry_mixin import RetryMixin
from nat.embedder.azure_openai_embedder import AzureOpenAIEmbedderModelConfig
from nat.embedder.nim_embedder import NIMEmbedderModelConfig
from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.embedder.huggingface_embedder import HuggingFaceEmbedderConfig
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


@register_embedder_client(config_type=AzureOpenAIEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def azure_openai_langchain(embedder_config: AzureOpenAIEmbedderModelConfig, builder: Builder):

    from langchain_openai import AzureOpenAIEmbeddings

    client = AzureOpenAIEmbeddings(
        **embedder_config.model_dump(exclude={"type", "api_version"},
                                     by_alias=True,
                                     exclude_none=True,
                                     exclude_unset=True),
        api_version=embedder_config.api_version,
    )

    if isinstance(embedder_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=embedder_config.num_retries,
                                  retry_codes=embedder_config.retry_on_status_codes,
                                  retry_on_messages=embedder_config.retry_on_errors)

    yield client


@register_embedder_client(config_type=NIMEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nim_langchain(embedder_config: NIMEmbedderModelConfig, builder: Builder):

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    client = NVIDIAEmbeddings(
        **embedder_config.model_dump(exclude={"type"}, by_alias=True, exclude_none=True, exclude_unset=True))

    if isinstance(embedder_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=embedder_config.num_retries,
                                  retry_codes=embedder_config.retry_on_status_codes,
                                  retry_on_messages=embedder_config.retry_on_errors)

    yield client


@register_embedder_client(config_type=OpenAIEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(embedder_config: OpenAIEmbedderModelConfig, builder: Builder):

    from langchain_openai import OpenAIEmbeddings

    client = OpenAIEmbeddings(
        **embedder_config.model_dump(exclude={"type"}, by_alias=True, exclude_none=True, exclude_unset=True))

    if isinstance(embedder_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=embedder_config.num_retries,
                                  retry_codes=embedder_config.retry_on_status_codes,
                                  retry_on_messages=embedder_config.retry_on_errors)

    yield client


@register_embedder_client(config_type=HuggingFaceEmbedderConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def huggingface_langchain(embedder_config: HuggingFaceEmbedderConfig, builder: Builder):
    """
    LangChain client for HuggingFace Embedder.
    
    Supports both local sentence-transformers and remote TEI/Inference API endpoints.
    """
    
    # Remote mode: use endpoint_url
    if embedder_config.endpoint_url:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        
        client = HuggingFaceEndpointEmbeddings(
            model=embedder_config.endpoint_url,
            huggingfacehub_api_token=str(embedder_config.api_key) if embedder_config.api_key else None,
        )
    else:
        # Local mode: use sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        
        model_kwargs = {
            "device": embedder_config.device,
        }
        
        if embedder_config.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        
        encode_kwargs = {
            "normalize_embeddings": embedder_config.normalize_embeddings,
        }
        
        if embedder_config.batch_size:
            encode_kwargs["batch_size"] = embedder_config.batch_size
        
        client = HuggingFaceEmbeddings(
            model_name=embedder_config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    if isinstance(embedder_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=embedder_config.num_retries,
            retry_codes=embedder_config.retry_on_status_codes,
            retry_on_messages=embedder_config.retry_on_errors
        )
    
    yield client
