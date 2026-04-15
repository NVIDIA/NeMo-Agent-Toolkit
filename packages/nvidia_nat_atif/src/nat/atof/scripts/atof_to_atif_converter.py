# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF-to-ATIF converter.

Converts a list of ATOF events (JSON-Lines wire format from NeMo-Flow
subscriber callbacks) into an ATIF Trajectory using NAT's native models.

Implements the same accumulator state machine as the Rust ``AtifExporter``
in ``crates/core/src/atif.rs``. See ``atof-event-format.md`` Section 7 for
the canonical mapping.

No NeMo-Flow or Harbor dependencies — uses ``nat.atof`` for input models
and ``nat.atif`` for output models.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from nat.atif.agent import Agent
from nat.atif.atif_step_extra import AtifAncestry
from nat.atif.atif_step_extra import AtifInvocationInfo
from nat.atif.observation import Observation
from nat.atif.observation_result import ObservationResult
from nat.atif.step import Step
from nat.atif.tool_call import ToolCall
from nat.atif.trajectory import Trajectory
from nat.atof.events import Event
from nat.atof.events import LLMEndEvent
from nat.atof.events import LLMStartEvent
from nat.atof.events import MarkEvent
from nat.atof.events import ToolEndEvent
from nat.atof.io import read_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_to_epoch(ts_str: str) -> float:
    """Convert ISO 8601 timestamp to epoch seconds, truncated to milliseconds."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return round(dt.timestamp(), 3)


def _build_ancestry(uuid: str, name: str, parent_uuid: str | None, name_map: dict[str, str]) -> dict:
    """Build ancestry dict from event identity and name lookup map."""
    return {
        "function_id": uuid,
        "function_name": name,
        "parent_id": parent_uuid or "",
        "parent_name": name_map.get(parent_uuid or "", "unknown"),
    }


def _build_invocation_info(start_ts: str | None, end_ts: str | None, invocation_id: str) -> dict:
    """Build invocation info dict from timestamps and ID."""
    info: dict = {
        "invocation_id": invocation_id,
        "framework": "nemo_flow",
        "status": "completed",
    }
    if start_ts:
        info["start_timestamp"] = _iso_to_epoch(start_ts)
    if end_ts:
        info["end_timestamp"] = _iso_to_epoch(end_ts)
    return info


def _extract_tool_calls(annotated_response: dict | None) -> list[dict]:
    """Extract tool call dicts from an annotated_response."""
    if not annotated_response:
        return []
    raw_calls = annotated_response.get("tool_calls") or []
    result = []
    for tc in raw_calls:
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        result.append({
            "tool_call_id": tc["id"],
            "function_name": tc["name"],
            "arguments": args,
        })
    return result


def _unwrap_llm_messages(input_obj: dict | None) -> list[dict]:
    """Extract messages from LLMStart input payload.

    Handles both NeMo-Flow envelope format (``{"content": {"messages": [...]}}``)
    and direct format (``{"messages": [...]}``).
    """
    if not input_obj:
        return []
    content = input_obj.get("content")
    if isinstance(content, dict):
        messages = content.get("messages", [])
        if messages:
            return messages
    messages = input_obj.get("messages", [])
    if messages:
        return messages
    return []


def _tool_call_order_key(sort_id: str, order_list: list[str]) -> int:
    """Return sort key for stable tool ordering by LLMEnd declaration order."""
    try:
        return order_list.index(sort_id)
    except ValueError:
        return len(order_list)


# ---------------------------------------------------------------------------
# Core accumulator
# ---------------------------------------------------------------------------


def _events_to_step_dicts(events: list[Event]) -> list[dict]:
    """Convert typed ATOF events to step dicts using the accumulator pattern.

    Mirrors the Rust AtifExporter processing loop (atof-event-format.md Section 7):
    - LLMStart → user step (messages from input)
    - LLMEnd → agent step (tool_calls from annotated_response)
    - ToolEnd → buffered observation, flushed on next LLMStart/Mark/end-of-stream
    - ScopeStart/ScopeEnd/ToolStart → skip (structural events)
    - Mark (with data) → system step
    """
    # Sort by timestamp
    sorted_events = sorted(events, key=lambda e: e.timestamp)

    # Pre-pass: build uuid → name and uuid → start_timestamp maps
    name_map: dict[str, str] = {}
    start_ts_map: dict[str, str] = {}

    for event in sorted_events:
        if event.uuid and event.name:
            name_map[event.uuid] = event.name
        if event.kind in ("LLMStart", "ToolStart", "ScopeStart"):
            start_ts_map[event.uuid] = event.timestamp

    # Accumulator state
    step_dicts: list[dict] = []
    pending_observations: list[dict] = []
    pending_obs_timestamp: str | None = None
    last_tool_call_map: dict[str, str] = {}
    last_tool_call_order: list[str] = []
    current_agent_step_idx: int | None = None
    pending_tool_ancestry: list[dict] = []
    pending_tool_invocations: list[dict] = []

    def flush_observations() -> None:
        nonlocal pending_observations, pending_obs_timestamp
        if not pending_observations:
            return
        sanitized = [{"content": obs["content"]} for obs in pending_observations]
        step_dicts.append({
            "source": "system",
            "message": "",
            "timestamp": pending_obs_timestamp,
            "observation": {"results": sanitized},
        })
        pending_observations = []
        pending_obs_timestamp = None

    def finalize_agent_extra() -> None:
        nonlocal current_agent_step_idx, pending_tool_ancestry, pending_tool_invocations
        if current_agent_step_idx is None:
            return
        step = step_dicts[current_agent_step_idx]
        extra = dict(step.get("extra", {}))
        if pending_tool_ancestry:
            sorted_anc = sorted(
                pending_tool_ancestry,
                key=lambda a: _tool_call_order_key(a.get("_sort_id", ""), last_tool_call_order),
            )
            for a in sorted_anc:
                a.pop("_sort_id", None)
            extra["tool_ancestry"] = sorted_anc
        if pending_tool_invocations:
            sorted_inv = sorted(
                pending_tool_invocations,
                key=lambda inv: _tool_call_order_key(inv.get("invocation_id", ""), last_tool_call_order),
            )
            extra["tool_invocations"] = sorted_inv
        step["extra"] = extra
        current_agent_step_idx = None
        pending_tool_ancestry = []
        pending_tool_invocations = []

    # Main event loop
    for event in sorted_events:
        if isinstance(event, LLMStartEvent):
            flush_observations()
            finalize_agent_extra()

            input_obj = event.input if isinstance(event.input, dict) else {}
            messages = _unwrap_llm_messages(input_obj)
            ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            message_str = json.dumps(messages, separators=(",", ":")) if messages else ""

            step_dicts.append({
                "source": "user",
                "message": message_str,
                "timestamp": event.timestamp,
                "extra": {"ancestry": ancestry},
            })

        elif isinstance(event, LLMEndEvent):
            flush_observations()

            ann_resp = (
                event.annotated_response.model_dump(exclude_none=True, mode="json")
                if event.annotated_response
                else None
            )
            tool_call_dicts = _extract_tool_calls(ann_resp)

            last_tool_call_map.clear()
            last_tool_call_order.clear()
            for tc in tool_call_dicts:
                last_tool_call_map[tc["function_name"]] = tc["tool_call_id"]
                last_tool_call_order.append(tc["tool_call_id"])

            ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            start_ts = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_ts, event.timestamp, event.uuid)

            extra: dict = {"ancestry": ancestry, "invocation": invocation}

            raw_output = event.output if isinstance(event.output, dict) else {}
            try:
                agent_msg = raw_output["choices"][0]["message"].get("content", "")
            except (KeyError, IndexError, TypeError):
                agent_msg = json.dumps(raw_output, separators=(",", ":")) if raw_output else ""

            step_dict: dict = {
                "source": "agent",
                "message": agent_msg or "",
                "timestamp": event.timestamp,
                "extra": extra,
            }
            if tool_call_dicts:
                step_dict["tool_calls"] = tool_call_dicts
            step_dicts.append(step_dict)
            current_agent_step_idx = len(step_dicts) - 1

        elif isinstance(event, ToolEndEvent):
            tool_call_id = event.tool_call_id
            if not tool_call_id:
                tool_call_id = last_tool_call_map.get(event.name)

            if pending_obs_timestamp is None:
                pending_obs_timestamp = event.timestamp

            output = event.output
            if isinstance(output, dict):
                content: str | None = json.dumps(output, separators=(",", ":"))
            elif isinstance(output, str):
                content = output
            elif output is not None:
                content = str(output)
            else:
                content = None

            pending_observations.append({"source_call_id": tool_call_id, "content": content})

            tool_ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            tool_ancestry["_sort_id"] = tool_call_id or ""
            pending_tool_ancestry.append(tool_ancestry)

            start_ts = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_ts, event.timestamp, tool_call_id or event.uuid)
            pending_tool_invocations.append(invocation)

        elif isinstance(event, MarkEvent) and event.data is not None:
            flush_observations()
            finalize_agent_extra()
            step_dicts.append({
                "source": "system",
                "message": json.dumps(event.data, separators=(",", ":")) if isinstance(event.data, dict) else str(event.data),
                "timestamp": event.timestamp,
            })

        else:
            logger.debug("Skipping %s event: %s", event.kind, event.name)

    # Finalize remaining state
    finalize_agent_extra()
    flush_observations()

    # Assign sequential step_ids
    for i, step in enumerate(step_dicts):
        step["step_id"] = i + 1

    return step_dicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert(events: list[Event]) -> Trajectory:
    """Convert a list of ATOF events to an ATIF Trajectory.

    Args:
        events: List of typed ATOF Event objects.

    Returns:
        A validated ATIF Trajectory.
    """
    step_dicts = _events_to_step_dicts(events)

    # Extract agent info from events
    agent_name = "unknown"
    model_name: str | None = None
    session_id = "atof-session"

    for event in events:
        if event.kind == "ScopeStart" and hasattr(event, "scope_type"):
            scope_type = getattr(event, "scope_type", "")
            if scope_type in ("agent", "Agent", "ScopeType.Agent"):
                agent_name = event.name
                break

    for event in events:
        if isinstance(event, LLMEndEvent):
            model_name = event.model_name or event.name
            break

    return Trajectory(
        session_id=session_id,
        agent=Agent(name=agent_name, version="1.0.0", model_name=model_name),
        steps=[Step(**sd) for sd in step_dicts],
    )


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Trajectory:
    """Read an ATOF JSON-Lines file and convert to an ATIF Trajectory.

    Args:
        input_path: Path to ``.jsonl`` file with ATOF events.
        output_path: Optional path to write the ATIF trajectory as JSON.

    Returns:
        The converted Trajectory.
    """
    events = read_jsonl(input_path)
    trajectory = convert(events)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traj_dict = trajectory.model_dump(exclude_none=True, mode="json")
        output_path.write_text(json.dumps(traj_dict, indent=2) + "\n")

    return trajectory
