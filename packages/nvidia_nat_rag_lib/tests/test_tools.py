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

import pytest
from nvidia_rag.rag_server.response_generator import Citations

from nat.builder.builder import Builder
from nat.builder.function import LambdaFunction
from nat.data_models.component_ref import FunctionGroupRef
from nat.plugins.rag_lib.client import NvidiaRAGFunctionGroup
from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
from nat.plugins.rag_lib.models import RAGGenerateResult
from nat.plugins.rag_lib.models import RAGSearchResult
from nat.plugins.rag_lib.tools.generate import RAGLibGenerateConfig
from nat.plugins.rag_lib.tools.generate import rag_lib_generate
from nat.plugins.rag_lib.tools.search import RAGLibSearchConfig
from nat.plugins.rag_lib.tools.search import rag_lib_search


class TestNvidiaRAGSearchTool:

    @pytest.fixture
    def mock_builder(self) -> MagicMock:
        return MagicMock(spec=Builder)

    @pytest.fixture
    def tool_config(self) -> RAGLibSearchConfig:
        return RAGLibSearchConfig(
            rag_client=FunctionGroupRef("rag_client"),
            collection_names=["test_collection"],
            reranker_top_k=5,
        )

    @pytest.fixture
    def function_group(self) -> NvidiaRAGFunctionGroup:
        config = NvidiaRAGLibConfig()
        group = NvidiaRAGFunctionGroup(config=config)
        group.rag_client = MagicMock()
        return group

    async def test_search_returns_results(self,
                                          tool_config: RAGLibSearchConfig,
                                          mock_builder: MagicMock,
                                          function_group: NvidiaRAGFunctionGroup) -> None:
        function_group.rag_client.search = AsyncMock(return_value=Citations(total_results=3, results=[]))
        mock_builder.get_function_group = AsyncMock(return_value=function_group)

        async with rag_lib_search(tool_config, mock_builder) as fn_info:
            tool = LambdaFunction.from_info(config=tool_config, info=fn_info, instance_name="search")
            result = await tool.acall_invoke(query="test query")

            assert isinstance(result, RAGSearchResult)
            assert result.citations.total_results == 3

    async def test_search_handles_error(self,
                                        tool_config: RAGLibSearchConfig,
                                        mock_builder: MagicMock,
                                        function_group: NvidiaRAGFunctionGroup) -> None:
        function_group.rag_client.search = AsyncMock(side_effect=Exception("Search failed"))
        mock_builder.get_function_group = AsyncMock(return_value=function_group)

        async with rag_lib_search(tool_config, mock_builder) as fn_info:
            tool = LambdaFunction.from_info(config=tool_config, info=fn_info, instance_name="search")
            result = await tool.acall_invoke(query="test query")

            assert isinstance(result, RAGSearchResult)
            assert result.citations.total_results == 0


class TestNvidiaRAGGenerateTool:

    @pytest.fixture
    def mock_builder(self) -> MagicMock:
        return MagicMock(spec=Builder)

    @pytest.fixture
    def tool_config(self) -> RAGLibGenerateConfig:
        return RAGLibGenerateConfig(
            rag_client=FunctionGroupRef("rag_client"),
            use_knowledge_base=True,
            enable_citations=True,
        )

    @pytest.fixture
    def function_group(self) -> NvidiaRAGFunctionGroup:
        config = NvidiaRAGLibConfig()
        group = NvidiaRAGFunctionGroup(config=config)
        group.rag_client = MagicMock()
        return group

    async def test_generate_returns_answer(self,
                                           tool_config: RAGLibGenerateConfig,
                                           mock_builder: MagicMock,
                                           function_group: NvidiaRAGFunctionGroup) -> None:

        async def mock_stream():
            yield 'data: {"id": "1", "model": "m", "choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"id": "1", "model": "m", "choices": [{"delta": {"content": " world"}}]}'
            yield 'data: [DONE]'

        function_group.rag_client.generate = AsyncMock(return_value=mock_stream())
        mock_builder.get_function_group = AsyncMock(return_value=function_group)

        async with rag_lib_generate(tool_config, mock_builder) as fn_info:
            tool = LambdaFunction.from_info(config=tool_config, info=fn_info, instance_name="generate")
            result = await tool.acall_invoke(query="test")

            assert isinstance(result, RAGGenerateResult)
            assert result.answer == "Hello world"

    async def test_generate_handles_error(self,
                                          tool_config: RAGLibGenerateConfig,
                                          mock_builder: MagicMock,
                                          function_group: NvidiaRAGFunctionGroup) -> None:
        function_group.rag_client.generate = AsyncMock(side_effect=Exception("Generate failed"))
        mock_builder.get_function_group = AsyncMock(return_value=function_group)

        async with rag_lib_generate(tool_config, mock_builder) as fn_info:
            tool = LambdaFunction.from_info(config=tool_config, info=fn_info, instance_name="generate")
            result = await tool.acall_invoke(query="test")

            assert isinstance(result, RAGGenerateResult)
            assert "Error generating answer" in result.answer

    async def test_generate_handles_empty_stream(self,
                                                 tool_config: RAGLibGenerateConfig,
                                                 mock_builder: MagicMock,
                                                 function_group: NvidiaRAGFunctionGroup) -> None:

        async def mock_empty_stream():
            yield 'data: [DONE]'

        function_group.rag_client.generate = AsyncMock(return_value=mock_empty_stream())
        mock_builder.get_function_group = AsyncMock(return_value=function_group)

        async with rag_lib_generate(tool_config, mock_builder) as fn_info:
            tool = LambdaFunction.from_info(config=tool_config, info=fn_info, instance_name="generate")
            result = await tool.acall_invoke(query="test")

            assert isinstance(result, RAGGenerateResult)
            assert result.answer == "No response generated."
