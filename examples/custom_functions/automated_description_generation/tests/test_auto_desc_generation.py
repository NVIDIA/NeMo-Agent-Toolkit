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

import pytest


@pytest.mark.asyncio
async def test_batch_summaries_are_reduced_from_parallel_writes():
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph
    from langgraph.types import Send

    from nat_automated_description_generation.utils.workflow_utils import BatchState
    from nat_automated_description_generation.utils.workflow_utils import OverallState

    async def create_batches(state: OverallState) -> dict[str, list[list[str]]]:
        return {"batches": [["doc1"], ["doc2"], ["doc3"]]}

    async def create_batch_summary(state: BatchState) -> dict[str, list[str]]:
        return {"summaries": [f"summary:{state['batch'][0]}"]}

    def dispatch_batches(state: OverallState) -> list[Send]:
        return [Send("create_batch_summary", {"batch": batch}) for batch in state["batches"]]

    graph = StateGraph(OverallState)
    graph.add_node("create_batches", create_batches)
    graph.add_node("create_batch_summary", create_batch_summary)
    graph.add_edge(START, "create_batches")
    graph.add_conditional_edges("create_batches", dispatch_batches, ["create_batch_summary"])
    graph.add_edge("create_batch_summary", END)

    result = await graph.compile().ainvoke({"contents": [], "summaries": []})

    assert sorted(result["summaries"]) == ["summary:doc1", "summary:doc2", "summary:doc3"]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "populate_milvus")
async def test_full_workflow(milvus_uri: str) -> None:
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_automated_description_generation.register import AutomatedDescriptionMilvusWorkflow

    config_file = locate_example_config(AutomatedDescriptionMilvusWorkflow)
    config = load_config(config_file)
    config.retrievers['retriever'].uri = HttpUrl(url=milvus_uri)

    # Unfortunately the workflow itself returns inconsistent results
    await run_workflow(config=config, question="List 5 subspecies of Aardvark?", expected_answer="Aardvark")
