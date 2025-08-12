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

import platform

import pytest
from nat_simple_web_query.register import WebQueryToolConfig

from nat.builder.workflow_builder import WorkflowBuilder
from nat.test.embedder import EmbedderTestConfig


@pytest.mark.skipif(platform.machine() == "aarch64",
                    reason="faiss not working on arm64 https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/72")
async def test_web_query_config():

    config = WebQueryToolConfig(webpage_url="https://www.google.com",
                                description="Test description",
                                chunk_size=1024,
                                embedder_name="web_embed")

    async with WorkflowBuilder() as builder:

        await builder.add_embedder("web_embed", config=EmbedderTestConfig())

        fn = await builder.add_function("webquery_tool", config)

        assert fn.config == config
        assert fn.description == config.description


@pytest.mark.skipif(platform.machine() == "aarch64",
                    reason="faiss not working on arm64 https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/72")
async def test_web_query_tool():

    config = WebQueryToolConfig(webpage_url="https://www.google.com",
                                description="Test description",
                                chunk_size=1024,
                                embedder_name="web_embed")

    async with WorkflowBuilder() as builder:

        await builder.add_embedder("web_embed", config=EmbedderTestConfig())

        fn = await builder.add_function("webquery_tool", config)

        result = await fn.ainvoke("search", to_type=str)

        assert "google" in result.lower()
        assert "search" in result.lower()
