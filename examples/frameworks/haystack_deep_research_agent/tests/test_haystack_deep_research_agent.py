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

import os
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture(name="opensearch_url", scope="session")
def opensearch_url_fixture(fail_missing: bool) -> str:
    url = os.getenv("NAT_CI_OPENSEARCH_URL", "http://localhost:9200")
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/_cluster/health", timeout=1) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        failure_reason = f"Unable to connect to open search server at {url}"
        if fail_missing:
            raise RuntimeError(failure_reason)
        pytest.skip(reason=failure_reason)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "serperdev")
async def test_full_workflow_e2e(opensearch_url: str) -> None:

    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    config_file = (Path(__file__).resolve().parents[1] / "src" / "nat_haystack_deep_research_agent" / "configs" /
                   "config.yml")

    config = load_config(config_file)
    config.workflow.opensearch_url = opensearch_url

    result = await run_workflow(question="Give a short overview of this workflow.",
                                expected_answer="workflow",
                                config=config)

    assert isinstance(result, str)
    assert len(result) > 0


def test_config_yaml_loads_and_has_keys() -> None:
    config_file = (Path(__file__).resolve().parents[1] / "configs" / "config.yml")

    with open(config_file, encoding="utf-8") as f:
        text = f.read()

    assert "workflow:" in text
    assert "_type: haystack_deep_research_agent" in text
    # key fields expected
    for key in [
            "llms:",
            "rag_llm:",
            "agent_llm:",
            "workflow:",
            "max_agent_steps:",
            "search_top_k:",
            "rag_top_k:",
            "opensearch_url:",
            "index_on_startup:",
            "data_dir:",
            "embedding_dim:",
    ]:
        assert key in text, f"Missing key: {key}"


def test_indexing_chunks_stay_within_embedder_token_limit() -> None:
    """Regression test for NAT-309.

    The indexing pipeline previously split by sentence (``split_length=10``), which could
    emit chunks larger than the NIM embedder 512-token limit and abort the entire indexing
    run. Guard the fix: bounded word-based chunking, plus ``truncate="END"`` on the embedder
    as a safety net. Runs fully offline (in-memory store, no embedding calls), so it needs
    neither OpenSearch nor a live NIM endpoint.
    """
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from nat_haystack_deep_research_agent.pipelines.indexing import _build_indexing_pipeline

    pipeline = _build_indexing_pipeline(InMemoryDocumentStore(), embedder_model="nvidia/nv-embedqa-e5-v5")
    components = pipeline.to_dict()["components"]

    # The embedder must truncate over-long inputs instead of failing the run.
    assert components["embedder"]["init_parameters"]["truncate"] == "END"

    # Chunking must be word-bounded (not sentence-based) so no chunk approaches the limit.
    splitter_params = components["splitter"]["init_parameters"]
    assert splitter_params["split_by"] == "word"
    split_length = splitter_params["split_length"]

    # A long, low-punctuation passage: sentence-based chunking grouped this into chunks far
    # larger than the limit, while word-based chunking keeps every chunk bounded.
    splitter = pipeline.get_component("splitter")
    splitter.warm_up()
    long_sentence = " ".join(f"token{i}" for i in range(30)) + "."
    document = Document(content=" ".join(long_sentence for _ in range(20)))
    chunks = splitter.run(documents=[document])["documents"]

    assert len(chunks) > 1, "expected the long document to be split into multiple chunks"
    for chunk in chunks:
        assert len(chunk.content.split()) <= split_length
