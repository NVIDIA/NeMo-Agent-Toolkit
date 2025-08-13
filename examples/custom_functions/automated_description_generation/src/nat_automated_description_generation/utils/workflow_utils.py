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

import asyncio
import math
from typing import Any
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send


# Define OverallState and BatchState for type annotations
class OverallState(TypedDict):
    contents: list[str]
    batches: list[list[str]]
    summaries: list[Any]
    collapsed_summaries: list[Document]
    final_summary: str
    bypass_map_reduce: bool


class BatchState(TypedDict):
    batch: list[str]


class SummarizationWorkflow:

    def __init__(
        self,
        llm,
        direct_summary_prompt: ChatPromptTemplate,
        map_prompt: ChatPromptTemplate,
        reduce_prompt: ChatPromptTemplate,
        max_token: int,
        batch_size: int,
    ):
        self.llm = llm
        self.direct_summary_prompt = direct_summary_prompt
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.max_token = max_token
        self.batch_size = batch_size

    def get_num_tokens_for_strings(self, contents: list[str]) -> int:
        return sum(self.llm.get_num_tokens(content) for content in contents)

    async def create_direct_summary(self, state: OverallState) -> dict[str, Any]:
        all_content = "\n\n---\n\n".join(state["contents"])
        prompt = self.direct_summary_prompt.invoke({"documents": all_content})
        response = await self.llm.ainvoke(prompt)
        return {"final_summary": response.content}

    def create_batches(self, state: OverallState) -> dict[str, Any]:
        total_tokens = self.get_num_tokens_for_strings(state["contents"])
        if total_tokens <= self.max_token:
            return {"bypass_map_reduce": True}
        avg_tokens_per_doc = total_tokens / len(state["contents"])
        target_batch_size = min(self.batch_size, max(1, math.floor(0.7 * self.max_token / avg_tokens_per_doc)))
        batches = []
        for i in range(0, len(state["contents"]), target_batch_size):
            batches.append(state["contents"][i:i + target_batch_size])
        return {"batches": batches, "bypass_map_reduce": False}

    async def create_batch_summary(self, state: BatchState) -> dict[str, Any]:
        combined_content = "\n\n---\n\n".join(state["batch"])
        prompt = self.map_prompt.invoke({"documents": combined_content})
        response = await self.llm.ainvoke(prompt)
        return {"summaries": [response.content]}

    def collect_batch_summaries(self, state: OverallState) -> dict[str, Any]:
        return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}

    async def _reduce_step(self, input_prompt: dict) -> str:
        prompt = self.reduce_prompt.invoke(input_prompt)
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def merge_local_summaries(self, state: OverallState) -> dict[str, Any]:
        doc_contents = [doc.page_content for doc in state["collapsed_summaries"]]
        doc_batches = []
        current_batch = []
        current_tokens = 0
        for doc_content in doc_contents:
            doc_tokens = self.get_num_tokens_for_strings([doc_content])
            if current_tokens + doc_tokens > self.max_token and current_batch:
                doc_batches.append([Document(page_content=content) for content in current_batch])
                current_batch = [doc_content]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc_content)
                current_tokens += doc_tokens
        if current_batch:
            doc_batches.append([Document(page_content=content) for content in current_batch])

        async def process_batch(doc_list):
            docs_content = "\n\n".join([doc.page_content for doc in doc_list])
            reduced_content = await self._reduce_step({"documents": docs_content})
            return Document(page_content=reduced_content)

        tasks = [process_batch(doc_list) for doc_list in doc_batches]
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
        return {"collapsed_summaries": results}

    def should_collapse(self, state: OverallState) -> str:
        doc_contents = [doc.page_content for doc in state["collapsed_summaries"]]
        num_tokens = self.get_num_tokens_for_strings(doc_contents)
        if num_tokens > self.max_token:
            return "merge_local_summaries"
        return "create_full_summary"

    def check_bypass(self, state: OverallState) -> str:
        if state.get("bypass_map_reduce"):
            return "create_direct_summary"
        return "map_batch_summaries"

    async def create_full_summary(self, state: OverallState) -> dict[str, Any]:
        doc_contents = "\n\n".join([doc.page_content for doc in state["collapsed_summaries"]])
        response = await self._reduce_step({"documents": doc_contents})
        return {"final_summary": response}

    async def map_batch_summaries(self, state: OverallState):
        for batch in state["batches"]:
            yield Send("create_batch_summary", {"batch": batch})
