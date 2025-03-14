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

from pydantic import ConfigDict

from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_embedder_client
from aiq.cli.register_workflow import register_embedder_provider
from aiq.data_models.embedder import EmbedderBaseConfig


class EmbedderTestConfig(EmbedderBaseConfig, name="test_embedder"):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "nvidia/nv-embedqa-e5-v5"
    embedding_size: int = 768


@register_embedder_provider(config_type=EmbedderTestConfig)
async def embedder_test_provider(config: EmbedderTestConfig, builder: Builder):

    yield EmbedderProviderInfo(config=config, description="Test embedder provider")


@register_embedder_client(config_type=EmbedderTestConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def embedder_langchain_test_client(config: EmbedderTestConfig, builder: Builder):

    from langchain.embeddings import DeterministicFakeEmbedding

    yield DeterministicFakeEmbedding(size=config.embedding_size)
