# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_embedder_client
from aiq.embedder.nim_embedder import NIMEmbedderModelConfig


@register_embedder_client(config_type=NIMEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nim_langchain(embedder_config: NIMEmbedderModelConfig, builder: Builder):

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    yield NVIDIAEmbeddings(**embedder_config.model_dump(exclude={"type"}, by_alias=True))


@register_embedder_client(config_type=NIMEmbedderModelConfig, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
async def nim_llamaindex(embedder_config: NIMEmbedderModelConfig, builder: Builder):

    from llama_index.embeddings.nvidia import NVIDIAEmbedding  # pylint: disable=no-name-in-module

    config_obj = {
        **embedder_config.model_dump(exclude={"type", "model_name"}, by_alias=True),
        "model":
            embedder_config.model_name,
    }

    yield NVIDIAEmbedding(**config_obj)
