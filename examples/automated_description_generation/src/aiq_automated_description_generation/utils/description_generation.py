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

import logging

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from .workflow_utils import OverallState

logger = logging.getLogger(__name__)


async def generate_description(collection_name: str, top_k, field_name, vectorstore, workflow) -> str:
    try:

        logger.debug("Getting Sample Chunks")
        schema = vectorstore.client.describe_collection(collection_name=collection_name)["fields"]
        dim = None

        for field in schema:
            if field["name"] == field_name:
                dim = field["params"]["dim"]
        if not dim:
            raise ValueError(f"Unable to get dimension for vector field: {field_name}.")

        dummy_vector = [0] * dim

        documents = vectorstore.similarity_search_by_vector(dummy_vector, k=top_k)

        logger.debug("Running Summarization Workflow")
        initial_state = {
            "contents": [doc.page_content for doc in documents],
            "batches": [],
            "summaries": [],
            "collapsed_summaries": [],
            "final_summary": "",
            "bypass_map_reduce": False
        }

        graph = StateGraph(OverallState)
        graph.add_node("create_batches", workflow.create_batches)
        graph.add_node("create_direct_summary", workflow.create_direct_summary)
        graph.add_node("create_batch_summary", workflow.create_batch_summary)
        graph.add_node("collect_batch_summaries", workflow.collect_batch_summaries)
        graph.add_node("merge_local_summaries", workflow.merge_local_summaries)
        graph.add_node("create_full_summary", workflow.create_full_summary)
        graph.add_node("map_batch_summaries", workflow.map_batch_summaries)

        graph.add_edge(START, "create_batches")
        graph.add_conditional_edges(
            "create_batches",
            workflow.check_bypass, {
                "create_direct_summary": "create_direct_summary", "map_batch_summaries": "map_batch_summaries"
            })
        graph.add_edge("create_batch_summary", "collect_batch_summaries")
        graph.add_conditional_edges("collect_batch_summaries", workflow.should_collapse)
        graph.add_conditional_edges("merge_local_summaries", workflow.should_collapse)
        graph.add_edge("create_full_summary", END)
        graph.add_edge("create_direct_summary", END)

        final_graph = graph.compile()
        final_summary = None

        async for step in final_graph.astream(initial_state, {"recursion_limit": 15}):
            if 'create_direct_summary' in step and not final_summary:
                final_summary = step['create_direct_summary']['final_summary']
            if 'create_full_summary' in step and not final_summary:
                final_summary = step['create_full_summary']['final_summary']

        if final_summary is not None:
            return final_summary

        return "There was an error generating a description for the collection."

    except Exception as e:
        logger.error("An error occurred when running the agent: %s", e)
        return "There was an error generating a description for the collection."
    finally:
        logger.debug("Finished summarization agent execution")
