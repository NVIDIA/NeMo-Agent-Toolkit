# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ATIF-native DataFrame creation — produces profiler DataFrame directly from Trajectory."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import pandas as pd

from nat.data_models.atif import AtifAncestry
from nat.data_models.atif import AtifStepExtra
from nat.data_models.atif import Trajectory
from nat.data_models.intermediate_step import IntermediateStepType


def _iso_to_epoch(iso_str: str | None) -> float:
    """Convert ISO 8601 timestamp to Unix epoch seconds."""
    if not iso_str:
        return 0.0
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _message_to_str(message: str | list | None) -> str:
    """Extract plain text from ATIF message (str or list of ContentPart)."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for part in message:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return " ".join(parts)
    return str(message)


def _ancestry_from_extra(step: Any, tool_index: int | None = None) -> dict[str, Any]:
    """Extract profiling ancestry from step.extra. Returns defaults for missing fields."""
    step_extra = AtifStepExtra.model_validate(getattr(step, "extra"))

    def to_flat(ancestry: AtifAncestry) -> dict[str, Any]:
        return {
            "function_id": ancestry.function_ancestry.function_id,
            "function_name": ancestry.function_ancestry.function_name,
            "parent_function_id": ancestry.function_ancestry.parent_id or "",
            "parent_function_name": ancestry.function_ancestry.parent_name or "",
            "span_event_timestamp": ancestry.span_event_timestamp,
            "framework": ancestry.framework,
        }

    if tool_index is not None:
        tool_ancestry = step_extra.tool_ancestry
        if tool_index < len(tool_ancestry):
            return to_flat(tool_ancestry[tool_index])
    return to_flat(step_extra.ancestry)


def _observation_content_to_str(content: Any) -> str:
    """Extract plain text from ATIF ObservationResult content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return " ".join(parts)
    return str(content)


def _row(
    *,
    event_timestamp: float,
    example_number: int,
    event_type: IntermediateStepType,
    span_event_timestamp: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    llm_text_input: str = "",
    llm_text_output: str = "",
    llm_new_token: str = "",
    llm_name: str = "",
    tool_name: str = "",
    function_name: str = "root",
    function_id: str = "root",
    parent_function_name: str = "",
    parent_function_id: str = "",
    row_uuid: str = "",
    framework: str | None = None,
) -> dict[str, Any]:
    """Build a DataFrame row dict matching create_standardized_dataframe schema."""
    return {
        "event_timestamp": event_timestamp,
        "example_number": example_number,
        "event_type": event_type,
        "span_event_timestamp": span_event_timestamp,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_text_input": llm_text_input,
        "llm_text_output": llm_text_output,
        "llm_new_token": llm_new_token,
        "llm_name": llm_name,
        "tool_name": tool_name,
        "function_name": function_name,
        "function_id": function_id,
        "parent_function_name": parent_function_name,
        "parent_function_id": parent_function_id,
        "UUID": row_uuid,
        "framework": framework,
    }


def create_dataframe_from_atif(trajectories: list[Trajectory]) -> pd.DataFrame:
    """
    Produce profiler DataFrame directly from ATIF trajectories.

    Same schema as create_standardized_dataframe. Option C: profiler consumes
    ATIF (Trajectory) internally; no intermediate ProfilerStep.
    """
    all_rows: list[dict[str, Any]] = []

    for example_number, trajectory in enumerate(trajectories):
        last_user_message = ""

        for step in trajectory.steps:
            ts = _iso_to_epoch(step.timestamp)
            msg = _message_to_str(step.message)

            if step.source == "user":
                last_user_message = msg
                anc = _ancestry_from_extra(step)
                all_rows.append(
                    _row(
                        event_timestamp=ts,
                        example_number=example_number,
                        event_type=IntermediateStepType.WORKFLOW_START,
                        llm_text_input=msg,
                        function_id=anc["function_id"],
                        function_name=anc["function_name"],
                        parent_function_id=anc["parent_function_id"],
                        parent_function_name=anc["parent_function_name"],
                        span_event_timestamp=anc.get("span_event_timestamp"),
                        framework=anc.get("framework"),
                    ))
                continue

            if step.source == "agent":
                step_anc = _ancestry_from_extra(step)

                # 1) Emit TOOL_END per tool call
                if step.tool_calls and step.observation:
                    obs_by_id = {r.source_call_id: r for r in step.observation.results if r.source_call_id}
                    for i, tc in enumerate(step.tool_calls):
                        obs = obs_by_id.get(tc.tool_call_id)
                        content = _observation_content_to_str(obs.content) if obs else ""
                        tool_anc = _ancestry_from_extra(step, tool_index=i)
                        all_rows.append(
                            _row(
                                event_timestamp=ts,
                                example_number=example_number,
                                event_type=IntermediateStepType.TOOL_END,
                                tool_name=tc.function_name,
                                llm_text_output=content,
                                row_uuid=str(uuid.uuid4()),
                                function_id=tool_anc["function_id"],
                                function_name=tool_anc["function_name"],
                                parent_function_id=tool_anc["parent_function_id"],
                                parent_function_name=tool_anc["parent_function_name"],
                                span_event_timestamp=tool_anc.get("span_event_timestamp"),
                                framework=tool_anc.get("framework"),
                            ))

                # 2) Emit LLM_START/LLM_END if metrics are present, even when all token counts are zero.
                if step.metrics is not None:
                    step_uuid = str(uuid.uuid4())
                    prompt = step.metrics.prompt_tokens or 0
                    completion = step.metrics.completion_tokens or 0
                    cached = step.metrics.cached_tokens or 0
                    total = prompt + completion + cached
                    llm_start_ts = (step_anc["span_event_timestamp"]
                                    if step_anc.get("span_event_timestamp") is not None else ts)
                    all_rows.append(
                        _row(
                            event_timestamp=llm_start_ts,
                            example_number=example_number,
                            event_type=IntermediateStepType.LLM_START,
                            llm_text_input=last_user_message,
                            llm_name=step.model_name or "",
                            row_uuid=step_uuid,
                            function_id=step_anc["function_id"],
                            function_name=step_anc["function_name"],
                            parent_function_id=step_anc["parent_function_id"],
                            parent_function_name=step_anc["parent_function_name"],
                            span_event_timestamp=step_anc.get("span_event_timestamp"),
                            framework=step_anc.get("framework"),
                        ))
                    all_rows.append(
                        _row(
                            event_timestamp=ts,
                            example_number=example_number,
                            event_type=IntermediateStepType.LLM_END,
                            llm_text_output=msg,
                            llm_name=step.model_name or "",
                            prompt_tokens=prompt,
                            completion_tokens=completion,
                            total_tokens=total,
                            row_uuid=step_uuid,
                            function_id=step_anc["function_id"],
                            function_name=step_anc["function_name"],
                            parent_function_id=step_anc["parent_function_id"],
                            parent_function_name=step_anc["parent_function_name"],
                            span_event_timestamp=step_anc.get("span_event_timestamp"),
                            framework=step_anc.get("framework"),
                        ))

                # 3) Emit WORKFLOW_END for message-only (no tools) final answer
                elif msg and not (step.tool_calls and step.observation):
                    all_rows.append(
                        _row(
                            event_timestamp=ts,
                            example_number=example_number,
                            event_type=IntermediateStepType.WORKFLOW_END,
                            llm_text_output=msg,
                            function_id=step_anc["function_id"],
                            function_name=step_anc["function_name"],
                            parent_function_id=step_anc["parent_function_id"],
                            parent_function_name=step_anc["parent_function_name"],
                            span_event_timestamp=step_anc.get("span_event_timestamp"),
                            framework=step_anc.get("framework"),
                        ))

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame.from_records(all_rows)


def _dict_to_step(d: dict[str, Any]) -> Any:
    """Convert step dict to object with attribute access for PredictionTrieBuilder."""
    from types import SimpleNamespace

    def wrap(v: Any) -> Any:
        if isinstance(v, dict):
            return SimpleNamespace(**{k: wrap(v2) for k, v2 in v.items()})
        return v

    return SimpleNamespace(**{k: wrap(v) for k, v in d.items()})


def dataframe_to_profiler_traces(df: pd.DataFrame) -> tuple[list[list[dict[str, Any]]], list[list[Any]]]:
    """
    Convert DataFrame to traces for JSON output and PredictionTrieBuilder.

    Returns (traces_dict, traces_obj):
    - traces_dict: list of list of step dicts for all_requests_data JSON
    - traces_obj: list of list of step-like objects for PredictionTrieBuilder.add_trace
    """
    if df.empty:
        return [], []

    traces_dict: list[list[dict[str, Any]]] = []
    traces_obj: list[list[Any]] = []
    for _, group in df.groupby("example_number", sort=True):
        steps_dict = []
        for _, row in group.iterrows():
            et = row.get("event_type")
            fn_name = str(row.get("function_name") or "root")
            parent_fn_name = str(row.get("parent_function_name") or "")
            step_dict = {
                "event_timestamp": float(row.get("event_timestamp", 0)),
                "span_event_timestamp": row.get("span_event_timestamp"),
                "event_type": et,
                "framework": row.get("framework"),
                "payload": {
                    "UUID": str(row.get("UUID", "") or "")
                },
                "token_usage": {
                    "prompt_tokens": row.get("prompt_tokens") or 0,
                    "completion_tokens": row.get("completion_tokens") or 0,
                    "total_tokens": row.get("total_tokens") or 0,
                },
                "llm_text_input": str(row.get("llm_text_input") or ""),
                "llm_text_output": str(row.get("llm_text_output") or ""),
                "llm_text_chunk": str(row.get("llm_new_token") or ""),
                "llm_name": str(row.get("llm_name") or ""),
                "tool_name": str(row.get("tool_name") or ""),
                "function_name": fn_name,
                "function_id": str(row.get("function_id") or "root"),
                "parent_function_name": parent_fn_name,
                "parent_function_id": str(row.get("parent_function_id") or ""),
                "function_ancestry": {
                    "function_name": fn_name,
                    "function_id": str(row.get("function_id") or "root"),
                    "parent_name": parent_fn_name,
                    "parent_id": str(row.get("parent_function_id") or ""),
                },
                "usage_info": {
                    "token_usage": {
                        "prompt_tokens": row.get("prompt_tokens") or 0,
                        "completion_tokens": row.get("completion_tokens") or 0,
                        "total_tokens": row.get("total_tokens") or 0,
                    },
                },
            }
            steps_dict.append(step_dict)
        traces_dict.append(steps_dict)
        traces_obj.append([_dict_to_step(s) for s in steps_dict])
    return traces_dict, traces_obj
