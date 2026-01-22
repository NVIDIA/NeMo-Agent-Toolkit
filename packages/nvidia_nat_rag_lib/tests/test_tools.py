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
from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
from nat.plugins.rag_lib.client import nvidia_rag_lib
from nat.plugins.rag_lib.models import RAGGenerateResult
from nat.plugins.rag_lib.models import RAGSearchResult


class TestNvidiaRAGLib:

    @pytest.fixture
    def mock_builder(self) -> MagicMock:
        builder = MagicMock(spec=Builder)
        builder.get_llm_config = MagicMock(return_value=None)
        builder.get_embedder_config = MagicMock(return_value=None)
        builder.get_retriever_config = AsyncMock(return_value=None)
        return builder

    @pytest.fixture
    def config(self) -> NvidiaRAGLibConfig:
        return NvidiaRAGLibConfig(collection_names=["test_collection"], )

    @pytest.fixture
    def mock_rag_client(self) -> MagicMock:
        client = MagicMock()
        client.search = AsyncMock(return_value=Citations(total_results=3, results=[]))
        return client

    async def test_search_returns_results(self,
                                          config: NvidiaRAGLibConfig,
                                          mock_builder: MagicMock,
                                          mock_rag_client: MagicMock) -> None:
        with patch("nvidia_rag.NvidiaRAG", return_value=mock_rag_client):
            async with nvidia_rag_lib(config, mock_builder) as group:
                functions = await group.get_all_functions()
                search_fn = next((f for name, f in functions.items() if name.endswith("search")), None)
                assert search_fn is not None

                result = await search_fn.acall_invoke(query="test query")

                assert isinstance(result, RAGSearchResult)
                assert result.citations.total_results == 3

    async def test_search_handles_error(self,
                                        config: NvidiaRAGLibConfig,
                                        mock_builder: MagicMock,
                                        mock_rag_client: MagicMock) -> None:
        mock_rag_client.search = AsyncMock(side_effect=Exception("Search failed"))

        with patch("nvidia_rag.NvidiaRAG", return_value=mock_rag_client):
            async with nvidia_rag_lib(config, mock_builder) as group:
                functions = await group.get_all_functions()
                search_fn = next((f for name, f in functions.items() if name.endswith("search")), None)
                result = await search_fn.acall_invoke(query="test query")

                assert isinstance(result, RAGSearchResult)
                assert result.citations.total_results == 0

    async def test_generate_returns_answer(self,
                                           config: NvidiaRAGLibConfig,
                                           mock_builder: MagicMock,
                                           mock_rag_client: MagicMock) -> None:

        async def mock_stream():
            yield 'data: {"id": "1", "model": "m", "choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"id": "1", "model": "m", "choices": [{"delta": {"content": " world"}}]}'
            yield 'data: [DONE]'

        mock_rag_client.generate = AsyncMock(return_value=mock_stream())

        with patch("nvidia_rag.NvidiaRAG", return_value=mock_rag_client):
            async with nvidia_rag_lib(config, mock_builder) as group:
                functions = await group.get_all_functions()
                generate_fn = next((f for name, f in functions.items() if name.endswith("generate")), None)
                assert generate_fn is not None

                result = await generate_fn.acall_invoke(query="test")

                assert isinstance(result, RAGGenerateResult)
                assert result.answer == "Hello world"

    async def test_generate_handles_error(self,
                                          config: NvidiaRAGLibConfig,
                                          mock_builder: MagicMock,
                                          mock_rag_client: MagicMock) -> None:
        mock_rag_client.generate = AsyncMock(side_effect=Exception("Generate failed"))

        with patch("nvidia_rag.NvidiaRAG", return_value=mock_rag_client):
            async with nvidia_rag_lib(config, mock_builder) as group:
                functions = await group.get_all_functions()
                generate_fn = next((f for name, f in functions.items() if name.endswith("generate")), None)
                result = await generate_fn.acall_invoke(query="test")

                assert isinstance(result, RAGGenerateResult)
                assert "Error generating answer" in result.answer

    async def test_generate_handles_empty_stream(self,
                                                 config: NvidiaRAGLibConfig,
                                                 mock_builder: MagicMock,
                                                 mock_rag_client: MagicMock) -> None:

        async def mock_empty_stream():
            yield 'data: [DONE]'

        mock_rag_client.generate = AsyncMock(return_value=mock_empty_stream())

        with patch("nvidia_rag.NvidiaRAG", return_value=mock_rag_client):
            async with nvidia_rag_lib(config, mock_builder) as group:
                functions = await group.get_all_functions()
                generate_fn = next((f for name, f in functions.items() if name.endswith("generate")), None)
                result = await generate_fn.acall_invoke(query="test")

                assert isinstance(result, RAGGenerateResult)
                assert result.answer == "No response generated."

    async def test_group_exposes_both_tools(self,
                                            config: NvidiaRAGLibConfig,
                                            mock_builder: MagicMock,
                                            mock_rag_client: MagicMock) -> None:
        with patch("nvidia_rag.NvidiaRAG", return_value=mock_rag_client):
            async with nvidia_rag_lib(config, mock_builder) as group:
                functions = await group.get_all_functions()
                function_names = list(functions.keys())
                assert any(name.endswith("search") for name in function_names)
                assert any(name.endswith("generate") for name in function_names)
