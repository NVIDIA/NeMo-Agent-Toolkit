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

from pathlib import Path

import pytest


async def _run_simple_rag_workflow(milvus_uri: str,
                                   config_file: Path,
                                   question="How do I install CUDA?",
                                   expected_answer="CUDA") -> str:
    """
    The tests/running of the workflow is the same for all the different configurations.
    However the API keys required are different.
    """
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    config = load_config(config_file)
    config.retrievers['cuda_retriever'].uri = HttpUrl(url=milvus_uri)
    config.retrievers['mcp_retriever'].uri = HttpUrl(url=milvus_uri)

    return await run_workflow(config=config, question=question, expected_answer=expected_answer)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "populate_milvus")
async def test_full_workflow(milvus_uri: str, examples_dir: Path):
    config_file = examples_dir / "RAG" / "simple_rag" / "configs" / "milvus_rag_config.yml"
    await _run_simple_rag_workflow(milvus_uri=milvus_uri, config_file=config_file)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "populate_milvus")
async def test_full_workflow_ttc(milvus_uri: str, examples_dir: Path):
    config_file = examples_dir / "RAG" / "simple_rag" / "configs" / "milvus_rag_config_ttc.yml"
    await _run_simple_rag_workflow(milvus_uri=milvus_uri, config_file=config_file)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "mem0_api_key", "populate_milvus")
async def test_full_workflow_memory(milvus_uri: str, examples_dir: Path):
    config_file = examples_dir / "RAG" / "simple_rag" / "configs" / "milvus_memory_rag_config.yml"
    await _run_simple_rag_workflow(milvus_uri=milvus_uri, config_file=config_file)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "tavily_api_key", "populate_milvus")
async def test_full_workflow_tools(milvus_uri: str, examples_dir: Path):
    config_file = examples_dir / "RAG" / "simple_rag" / "configs" / "milvus_rag_tools_config.yml"
    await _run_simple_rag_workflow(milvus_uri=milvus_uri, config_file=config_file)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "mem0_api_key", "tavily_api_key", "populate_milvus")
async def test_full_workflow_memory_tools(milvus_uri: str, examples_dir: Path):
    config_file = examples_dir / "RAG" / "simple_rag" / "configs" / "milvus_memory_rag_tools_config.yml"
    await _run_simple_rag_workflow(milvus_uri=milvus_uri, config_file=config_file)
