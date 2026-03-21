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

import ast
from collections import defaultdict
from typing import Any

from nat.data_models.atif.observation_result import ObservationResult
from nat.data_models.atif.step import Step
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.evaluator.atif_base_evaluator import AtifBaseEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from .models import _ToolCall
from .models import _ToolFailureReasoning
from .models import _ToolSummary


class ToolFailureEvaluator(BaseEvaluator, AtifBaseEvaluator):
    """Evaluates tool call success rate by checking ``status="error"`` on TOOL_END events.

    Args:
        max_concurrency: Maximum number of items to evaluate concurrently.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating tool failures")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Evaluate a single item's legacy trajectory for tool failures."""
        if not item.trajectory:
            return EvalOutputItem(id=item.id, score=1.0, reasoning=_ToolFailureReasoning())

        total_tool_calls: int = 0
        failed_tool_calls: int = 0
        calls_by_tool: defaultdict[str, list[_ToolCall]] = defaultdict(list)

        for step in item.trajectory:
            if step.event_type != IntermediateStepType.TOOL_END:
                continue
            tool_name: str = step.name or "unknown"
            tool_output: dict[str, Any] | None = step.data.output if step.data else None
            is_error: bool = self._is_tool_message_error(tool_output)

            call: _ToolCall = _ToolCall(
                input=self._extract_tool_message_input(step.data.input if step.data else None),
                output=self._extract_tool_message_content(tool_output) if not is_error else None,
                error=self._extract_tool_message_content(tool_output) if is_error else None,
            )
            calls_by_tool[tool_name].append(call)
            total_tool_calls += 1
            if is_error:
                failed_tool_calls += 1

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
        failed_tools: list[str] = [ts.tool_name for ts in per_tool_summary]
        reasoning: _ToolFailureReasoning = _ToolFailureReasoning(
            total_tool_calls=total_tool_calls,
            failed_tool_calls=failed_tool_calls,
            failed_tools=failed_tools,
            score=score,
            per_tool_summary=per_tool_summary,
        )
        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        """Evaluate a single ATIF sample for tool failures."""
        steps: list[Step] = sample.trajectory.steps if sample.trajectory else []

        total_tool_calls: int = 0
        failed_tool_calls: int = 0
        calls_by_tool: defaultdict[str, list[_ToolCall]] = defaultdict(list)

        for step in steps:
            if not step.tool_calls or step.source != "agent":
                continue

            observations: list[ObservationResult] = (step.observation.results if step.observation else [])

            for index, atif_tool_call in enumerate(step.tool_calls):
                observation_content: str = ""
                if index < len(observations) and observations[index].content:
                    raw_content: str | list[Any] | None = observations[index].content
                    observation_content = raw_content if isinstance(raw_content, str) else str(raw_content)

                is_error: bool = False
                error_content: str = ""

                # Check step.extra["tool_errors"] for structured error metadata
                extra_errors: list[dict[str, Any]] = (step.extra or {}).get("tool_errors", [])
                matching_extra: dict[str, Any] | None = None
                for tool_error_entry in extra_errors:
                    if tool_error_entry.get("tool") == atif_tool_call.function_name:
                        is_error = True
                        matching_extra = tool_error_entry
                        break

                # Parse observation content as a serialized ToolMessage dict
                if not is_error and observation_content:
                    parsed: dict[str, Any] | None = self._parse_tool_message_dict(observation_content)
                    if parsed is not None and parsed.get("status") == "error":
                        is_error = True
                        error_content = str(parsed.get("content", observation_content))

                # Match raw error patterns like "ValueError: ..." in observation content
                if not is_error and observation_content:
                    candidate_type: str = (observation_content.split(":", 1)[0].strip()
                                           if ":" in observation_content else "")
                    if candidate_type.isidentifier() and candidate_type.endswith("Error"):
                        is_error = True

                # Resolve error content from whichever source matched
                if is_error and not error_content:
                    if matching_extra:
                        error_content = matching_extra.get("error", observation_content)
                    else:
                        error_content = observation_content

                call: _ToolCall = _ToolCall(
                    input=atif_tool_call.arguments if atif_tool_call.arguments else None,
                    output=observation_content if not is_error else None,
                    error=error_content if is_error else None,
                )
                calls_by_tool[atif_tool_call.function_name].append(call)
                total_tool_calls += 1
                if is_error:
                    failed_tool_calls += 1

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
        failed_tools: list[str] = [ts.tool_name for ts in per_tool_summary]
        reasoning: _ToolFailureReasoning = _ToolFailureReasoning(
            total_tool_calls=total_tool_calls,
            failed_tool_calls=failed_tool_calls,
            failed_tools=failed_tools,
            score=score,
            per_tool_summary=per_tool_summary,
        )
        return EvalOutputItem(id=sample.item_id, score=score, reasoning=reasoning)

    def _success_rate(self, total: int, failed: int) -> float:
        """Compute success rate as a float in [0.0, 1.0]."""
        return (total - failed) / total if total > 0 else 1.0

    def _is_tool_message_error(self, tool_output: dict[str, Any] | None) -> bool:
        """Check whether a ToolMessage output indicates an error.

        Handles both live ToolMessage objects and their serialized dict form
        by checking for ``status='error'`` via attribute or key access.

        Args:
            tool_output: The ToolMessage or its dict serialization from a TOOL_END event.

        Returns:
            True if the tool output carries an error status.
        """
        if tool_output is None:
            return False
        status: str | None = (getattr(tool_output, "status", None)
                              or (tool_output.get("status") if isinstance(tool_output, dict) else None))
        return status == "error"

    def _extract_tool_message_content(self, tool_output: dict[str, Any] | None) -> str:
        """Extract the content string from a ToolMessage output.

        Handles both live ToolMessage objects and their serialized dict form
        by reading the ``content`` field via attribute or key access.

        Args:
            tool_output: The ToolMessage or its dict serialization from a TOOL_END event.

        Returns:
            The content string, or a stringified fallback if no content field is found.
        """
        if tool_output is None:
            return ""
        content: str | None = (getattr(tool_output, "content", None)
                               or (tool_output.get("content") if isinstance(tool_output, dict) else None))
        if content is not None:
            return str(content)
        return str(tool_output)

    def _extract_tool_message_input(
        self,
        tool_input: dict[str, Any] | str | None,
    ) -> dict[str, Any] | str | None:
        """Normalize ToolMessage input arguments for JSON serialization.

        Args:
            tool_input: The raw tool call arguments from a TOOL_END event payload.

        Returns:
            A dict, string, or None suitable for JSON serialization.
        """
        if tool_input is None:
            return None
        if isinstance(tool_input, dict):
            return tool_input
        raw_str: str = str(tool_input)
        if not raw_str:
            return None
        return raw_str

    def _parse_tool_message_dict(self, content: str) -> dict[str, Any] | None:
        """Parse a stringified ToolMessage dict from ATIF observation content.

        The ATIF converter serializes ToolMessage objects via ``_safe_str``,
        producing Python dict literals as strings.

        Args:
            content: Raw observation content that may contain a stringified dict.

        Returns:
            The parsed dict if successful, None otherwise.
        """
        if not content or not content.startswith("{"):
            return None
        try:
            parsed: dict[str, Any] | None = ast.literal_eval(content)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return None
