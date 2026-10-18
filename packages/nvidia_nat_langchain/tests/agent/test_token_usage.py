# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Sum the token usage a model reports over one agent run."""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult

from nat.plugins.langchain.agent.token_usage import TokenUsageCollector


def _reply(usage_metadata: dict | None) -> LLMResult:
    message = AIMessage(content="x", usage_metadata=usage_metadata) if usage_metadata else AIMessage(content="x")
    return LLMResult(generations=[[ChatGeneration(message=message)]])


async def test_collector_sums_every_reply():
    collector = TokenUsageCollector()
    await collector.on_llm_end(_reply({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}))
    await collector.on_llm_end(_reply({"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}))
    usage = collector.usage()
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (250, 50, 300)
    assert usage.prompt_tokens_details is None  # no reply reported cached tokens


async def test_collector_sums_cached_prompt_tokens():
    collector = TokenUsageCollector()
    await collector.on_llm_end(
        _reply({
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "input_token_details": {
                "cache_read": 40
            }
        }))
    await collector.on_llm_end(_reply({"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}))
    assert collector.usage().prompt_tokens_details.cached_tokens == 40


async def test_collector_reports_none_when_no_reply_reported_usage():
    collector = TokenUsageCollector()
    await collector.on_llm_end(_reply(None))
    assert collector.usage() is None
