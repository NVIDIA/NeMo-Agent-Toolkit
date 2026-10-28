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

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from nat.data_models.api_server import PromptTokensDetails
from nat.data_models.api_server import Usage
from nat.plugins.langchain.agent.tool_calling_agent.agent import extract_token_usage


class TokenUsageCollector(AsyncCallbackHandler):
    """Callback handler that adds up the `usage_metadata` of the agent's model replies in one run.

    Pass a new instance in the `callbacks` of one graph invocation. It counts the agent's own model calls; tools
    run with their own config, so LLM calls made inside a tool are not included. Replies that report no usage are
    skipped. `usage()` returns ``None`` when no reply reported usage, so a caller can tell that apart from a
    reported zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._cached_tokens = 0
        self._saw_usage = False
        self._saw_cached = False

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = extract_token_usage(message) if message is not None else None
                if usage is None:
                    continue
                self._prompt_tokens += usage["input_tokens"]
                self._completion_tokens += usage["output_tokens"]
                self._total_tokens += usage["total_tokens"]
                self._saw_usage = True
                cached = (usage.get("input_token_details") or {}).get("cache_read")
                if cached is not None:
                    self._cached_tokens += cached
                    self._saw_cached = True

    def usage(self) -> Usage | None:
        if not self._saw_usage:
            return None
        details = PromptTokensDetails(cached_tokens=self._cached_tokens) if self._saw_cached else None
        return Usage(prompt_tokens=self._prompt_tokens,
                     completion_tokens=self._completion_tokens,
                     total_tokens=self._total_tokens,
                     prompt_tokens_details=details)
