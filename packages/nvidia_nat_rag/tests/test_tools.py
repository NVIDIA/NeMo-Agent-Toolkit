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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nvidia_rag.rag_server.response_generator import Citations

from nat.builder.builder import Builder
from nat.plugins.rag.client import NATRAGConfig
from nat.plugins.rag.client import nat_rag
from nat.plugins.rag.models import RAGGenerateResult
from nat.plugins.rag.models import RAGSearchResult


class TestNATRAG:

    @pytest.fixture(name="mock_builder")
    def fixture_mock_builder(self) -> MagicMock:
        from pydantic import HttpUrl

        from nat.embedder.nim_embedder import NIMEmbedderModelConfig
        from nat.llm.nim_llm import NIMModelConfig
        from nat.retriever.milvus.register import MilvusRetrieverConfig

        builder = MagicMock(spec=Builder)
        builder.get_llm_config = MagicMock(return_value=NIMModelConfig(
            model_name="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
        ))
        builder.get_embedder_config = MagicMock(return_value=NIMEmbedderModelConfig(
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            base_url="https://integrate.api.nvidia.com/v1",
        ))
        builder.get_retriever_config = AsyncMock(return_value=MilvusRetrieverConfig(
            uri=HttpUrl("http://localhost:19530"),
            collection_name="test_collection",
            embedding_model="nim_embedder",
        ))
        return builder

    @pytest.fixture(name="config")
    def fixture_config(self) -> NATRAGConfig:
        from nat.data_models.component_ref import EmbedderRef
        from nat.data_models.component_ref import LLMRef
        from nat.data_models.component_ref import RetrieverRef
        return NATRAGConfig(
            llm=LLMRef("nim_llm"),
            embedder=EmbedderRef("nim_embedder"),
            retriever=RetrieverRef("cuda_retriever"),
            collection_names=["test_collection"],
        )

    @pytest.fixture(name="mock_rag_client")
    def fixture_mock_rag_client(self) -> MagicMock:
        client = MagicMock()
        client.search = AsyncMock(return_value=Citations(total_results=3, results=[]))
        return client

    async def test_search_returns_results(self,
                                          config: NATRAGConfig,
                                          mock_builder: MagicMock,
                                          mock_rag_client: MagicMock) -> None:
        with patch("nvidia_rag.rag_server.main.NvidiaRAG", return_value=mock_rag_client):
            async with nat_rag(config, mock_builder) as group:
                functions = await group.get_all_functions()
                search_fn = next((f for name, f in functions.items() if name.endswith("search")), None)
                assert search_fn is not None

                result = await search_fn.acall_invoke(query="test query")

                assert isinstance(result, RAGSearchResult)
                assert result.citations.total_results == 3

    async def test_generate_returns_answer(self,
                                           config: NATRAGConfig,
                                           mock_builder: MagicMock,
                                           mock_rag_client: MagicMock) -> None:

        async def mock_stream():
            yield 'data: {"id": "1", "model": "m", "choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"id": "1", "model": "m", "choices": [{"delta": {"content": " world"}}]}'
            yield 'data: [DONE]'

        mock_rag_client.generate = AsyncMock(return_value=mock_stream())

        with patch("nvidia_rag.rag_server.main.NvidiaRAG", return_value=mock_rag_client):
            async with nat_rag(config, mock_builder) as group:
                functions = await group.get_all_functions()
                generate_fn = next((f for name, f in functions.items() if name.endswith("generate")), None)
                assert generate_fn is not None

                result = await generate_fn.acall_invoke(query="test")

                assert isinstance(result, RAGGenerateResult)
                assert result.answer == "Hello world"

    async def test_generate_handles_empty_stream(self,
                                                 config: NATRAGConfig,
                                                 mock_builder: MagicMock,
                                                 mock_rag_client: MagicMock) -> None:

        async def mock_empty_stream():
            yield 'data: [DONE]'

        mock_rag_client.generate = AsyncMock(return_value=mock_empty_stream())

        with patch("nvidia_rag.rag_server.main.NvidiaRAG", return_value=mock_rag_client):
            async with nat_rag(config, mock_builder) as group:
                functions = await group.get_all_functions()
                generate_fn = next((f for name, f in functions.items() if name.endswith("generate")), None)
                result = await generate_fn.acall_invoke(query="test")

                assert isinstance(result, RAGGenerateResult)
                assert result.answer == "No response generated."

    async def test_group_exposes_both_tools(self,
                                            config: NATRAGConfig,
                                            mock_builder: MagicMock,
                                            mock_rag_client: MagicMock) -> None:
        with patch("nvidia_rag.rag_server.main.NvidiaRAG", return_value=mock_rag_client):
            async with nat_rag(config, mock_builder) as group:
                functions = await group.get_all_functions()
                function_names = list(functions.keys())
                assert any(name.endswith("search") for name in function_names)
                assert any(name.endswith("generate") for name in function_names)
