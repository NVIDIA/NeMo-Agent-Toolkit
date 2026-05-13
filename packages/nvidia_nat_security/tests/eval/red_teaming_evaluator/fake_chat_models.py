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

from __future__ import annotations

import json
from typing import Any
from typing import Union  # noqa: F401

from langchain_core.caches import BaseCache  # noqa: F401
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks import Callbacks  # noqa: F401
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult


class FakeJudgeLLM(BaseChatModel):
    """
    A deterministic mock LLM judge that evaluates outputs based on pattern matching.
    Returns scores based on the presence of specific patterns in the output.
    """

    # Define patterns and their scores as class attributes. Can be overridden in the constructor.
    patterns: dict[str, float] = {}

    def _evaluate_output(self, messages: list[BaseMessage]) -> AIMessage:
        """Extract and evaluate output from messages."""
        # Extract the prompt from messages
        prompt = ""
        for msg in messages:
            if hasattr(msg, "content"):
                prompt += str(msg.content)

        # Extract the generated output from the prompt
        generated_output = ""
        if "**System Output:**" in prompt:
            output_section = prompt.split("**System Output:**")[1]
            if "\n\n" in output_section:
                generated_output = output_section.split("\n\n")[0].strip()
            else:
                generated_output = output_section.strip()

        # Check for patterns (case-insensitive)
        generated_output_lower = generated_output.lower()
        max_score = 0.0
        matched_pattern = None

        for pattern, score in self.patterns.items():
            if pattern in generated_output_lower:
                if score > max_score:
                    max_score = score
                    matched_pattern = pattern

        # If no pattern matched, default to 0.0
        matched_pattern = "no pattern detected" if not matched_pattern else matched_pattern

        # Generate reasoning
        reasoning = f"Pattern '{matched_pattern}' detected in output. Score: {max_score}"

        # Return JSON response matching the expected format
        response_json = {"score": max_score, "reasoning": reasoning}

        return AIMessage(content=json.dumps(response_json))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate method required by BaseChatModel."""
        response = self._evaluate_output(messages)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync generate method required by BaseChatModel."""
        response = self._evaluate_output(messages)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-judge-llm"


# Rebuild the model to ensure Pydantic can properly validate it
# This is needed because BaseChatModel has forward references that need to be resolved
FakeJudgeLLM.model_rebuild()
