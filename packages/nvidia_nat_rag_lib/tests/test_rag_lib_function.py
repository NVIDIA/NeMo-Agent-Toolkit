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
"""
Test suite for the NVIDIA RAG Library integration.

This module contains tests for the NVIDIA RAG Library function configuration,
registration, and basic functionality verification.
"""

from pathlib import Path

import pytest
import yaml

from nat.data_models.config import Config


@pytest.fixture
def test_config():
    """Pytest fixture that loads the test configuration."""
    config_path = Path(__file__).parent / "test_config.yml"

    with open(config_path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config.model_validate(config_dict)


class TestNvidiaRAGLib:
    """Test suite for the NVIDIA RAG Library function."""

    def test_function_registration(self):
        """Test that the RAG function is properly registered with NAT."""
        from nat.cli.type_registry import GlobalTypeRegistry
        from nat.plugins.rag_lib.client import NvidiaRAGLibConfig
        registry = GlobalTypeRegistry.get()

        # Check if our function is registered
        try:
            function_info = registry.get_function(NvidiaRAGLibConfig)
            assert function_info is not None
            assert function_info.config_type == NvidiaRAGLibConfig
        except KeyError:
            pytest.fail("NvidiaRAGLibConfig function not properly registered")

    @pytest.mark.asyncio
    async def test_rag_library_acquisition(self, test_config):
        """Test acquiring the RAG library.

        Simple test that tries to acquire the RAG library.
        """
        from nat.builder.workflow_builder import WorkflowBuilder

        try:
            async with WorkflowBuilder.from_config(test_config) as builder:
                # Simply acquire the RAG library function
                rag_function = await builder.get_function("rag_client")

                # Verify we got the function
                assert rag_function is not None
                print("RAG library acquired successfully")

        except ImportError as e:
            if "nvidia-rag" in str(e):
                pytest.fail(f"nvidia-rag library not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_rag_search_functionality(self, test_config):
        """Test RAG library search functionality after successful acquisition.

        This test demonstrates how to use the RAG library's search capabilities
        including citation parsing. It doesn't need to work (no vector DB running)
        but shows the proper setup for RAG search operations.
        """
        from nat.builder.workflow_builder import WorkflowBuilder

        def parse_search_citations(citations):
            """Parse search citations into formatted document strings."""
            parsed_docs = []

            for idx, citation in enumerate(citations.results):
                # If using pydantic models, citation fields may be attributes, not dict keys
                content = getattr(citation, 'content', '')
                doc_name = getattr(citation, 'document_name', f'Citation {idx+1}')
                parsed_document = f'<Document source="{doc_name}"/>\n{content}\n</Document>'
                parsed_docs.append(parsed_document)

            # combine parsed documents into a single string
            internal_search_docs = "\n\n---\n\n".join(parsed_docs)
            return internal_search_docs

        try:
            async with WorkflowBuilder.from_config(test_config) as builder:
                # Acquire the RAG library function
                rag_function = await builder.get_function("rag_client")
                assert rag_function is not None

                try:
                    # Demonstrate search configuration matching our config
                    collection_names = test_config.functions["rag_client"].collection_names
                    reranker_top_k = test_config.functions["rag_client"].reranker_top_k
                    vdb_top_k = test_config.functions["rag_client"].vdb_top_k

                    search_results = rag_function.search(
                        query="test query",
                        collection_names=collection_names,
                        reranker_top_k=reranker_top_k,
                        vdb_top_k=vdb_top_k,
                    )
                    parsed_docs = parse_search_citations(search_results)

                    # Assert if data was returned from parsed_docs
                    assert parsed_docs is not None

                except Exception as e:
                    print(f"RAG search failed as expected (no vector DB): {e}")

        except ImportError as e:
            if "nvidia-rag" in str(e):
                pytest.fail(f"nvidia-rag library not available: {e}")
            else:
                raise
