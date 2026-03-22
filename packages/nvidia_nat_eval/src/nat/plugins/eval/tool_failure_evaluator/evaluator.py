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
"""Tool failure evaluator for agent trajectories."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import ToolErrorData
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from .models import _ToolCall
from .models import _ToolFailureReasoning
from .models import _ToolSummary


class ToolFailureEvaluator(BaseEvaluator):
    """Evaluates tool call success rate from IntermediateStep trajectories.

    Args:
        max_concurrency: Maximum number of items to evaluate concurrently.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating tool failures")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Evaluate a single item's IntermediateStep trajectory for tool failures."""
        if not item.trajectory:
            return EvalOutputItem(id=item.id, score=1.0, reasoning=_ToolFailureReasoning())

        total_tool_calls: int = 0
        failed_tool_calls: int = 0
        calls_by_tool: defaultdict[str, list[_ToolCall]] = defaultdict(list)

        for step in item.trajectory:
            if step.event_type != IntermediateStepType.TOOL_END:
                continue
            tool_name: str = step.name or "unknown"
            tool_output: Any = step.data.output if step.data else None
            is_error: bool = self._is_tool_error(tool_output)

            call: _ToolCall = _ToolCall(
                input=self._extract_tool_input(step.data.input if step.data else None),
                output=self._extract_tool_content(tool_output) if not is_error else None,
                error=self._extract_tool_content(tool_output) if is_error else None,
            )
            calls_by_tool[tool_name].append(call)
            total_tool_calls += 1
            if is_error:
                failed_tool_calls += 1

        return self._build_output(item.id, total_tool_calls, failed_tool_calls, calls_by_tool)

    def _build_output(
        self,
        item_id: Any,
        total_tool_calls: int,
        failed_tool_calls: int,
        calls_by_tool: defaultdict[str, list[_ToolCall]],
    ) -> EvalOutputItem:
        """Build the EvalOutputItem from aggregated tool call data."""
        score: float = self._success_rate(total_tool_calls, failed_tool_calls)
        per_tool_summary: list[_ToolSummary] = [
            _ToolSummary(
                tool_name=name,
                total_calls=len(attempts),
                failed_calls=failed_count,
                failed_attempts=[a for a in attempts if a.error is not None],
            ) for name, attempts in calls_by_tool.items()
            if (failed_count := sum(1 for a in attempts if a.error is not None)) > 0
        ]
        failed_tools: list[str] | None = [ts.tool_name for ts in per_tool_summary] if per_tool_summary else None
        reasoning: _ToolFailureReasoning = _ToolFailureReasoning(
            total_tool_calls=total_tool_calls,
            failed_tool_calls=failed_tool_calls,
            failed_tools=failed_tools,
            score=score,
            per_tool_summary=per_tool_summary if per_tool_summary else None,
        )
        return EvalOutputItem(id=item_id, score=score, reasoning=reasoning)

    def _success_rate(self, total: int, failed: int) -> float:
        """Compute success rate as a float in [0.0, 1.0]."""
        return (total - failed) / total if total > 0 else 1.0

    def _is_tool_error(self, tool_output: Any) -> bool:
        """Check whether a tool output indicates an error."""
        if tool_output is None:
            return False
        if isinstance(tool_output, ToolErrorData):
            return True
        if isinstance(tool_output, dict) and "error_type" in tool_output:
            return True
        return False

    def _extract_tool_content(self, tool_output: Any) -> str:
        """Extract the content string from a tool output."""
        if tool_output is None:
            return ""
        if isinstance(tool_output, ToolErrorData):
            return tool_output.content
        if isinstance(tool_output, dict):
            return str(tool_output.get("content", tool_output))
        return str(tool_output)

    def _extract_tool_input(self, tool_input: Any) -> dict[str, Any] | str | None:
        """Normalize tool input arguments for JSON serialization."""
        if tool_input is None:
            return None
        if isinstance(tool_input, dict):
            return tool_input
        raw_str: str = str(tool_input)
        if not raw_str:
            return None
        return raw_str
