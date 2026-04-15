# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF-to-ATIF converter.

Converts a list of ATOF events (JSON-Lines wire format from agent runtime
subscriber callbacks) into an ATIF Trajectory using NAT's native models.

Event model: 3 event kinds (ScopeStart / ScopeEnd / Mark) per spec v0.1.
Dispatch keys on ``(kind, scope_type)``. ``tool_call_id`` lives directly on
``ScopeStart``/``ScopeEnd`` for ``scope_type == "tool"`` events, so no
name-based fallback map is needed.

See ``atof-event-format.md`` §3 (event kinds), §4 (scope_type vocabulary),
§5 (status semantics), §6 (stream semantics) and the companion
``atof-to-atif-converter.md`` for the normative mapping.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nat.atif.agent import Agent
from nat.atif.step import Step
from nat.atif.trajectory import Trajectory
from nat.atof.events import Event
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEndEvent
from nat.atof.events import ScopeStartEvent
from nat.atof.io import read_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ancestry(uuid: str, name: str, parent_uuid: str | None, name_map: dict[str, str]) -> dict:
    """Build ancestry dict from event identity and name lookup map."""
    return {
        "function_id": uuid,
        "function_name": name,
        "parent_id": parent_uuid or "",
        "parent_name": name_map.get(parent_uuid or "", "unknown"),
    }


def _build_invocation_info(start_micros: int | None, end_micros: int | None, invocation_id: str) -> dict:
    """Build invocation info from microsecond timestamps + ID.

    Emits ATIF's seconds-as-float shape at millisecond precision.
    """
    info: dict = {
        "invocation_id": invocation_id,
        "framework": "nat",
        "status": "completed",
    }
    if start_micros is not None:
        info["start_timestamp"] = round(start_micros / 1_000_000, 3)
    if end_micros is not None:
        info["end_timestamp"] = round(end_micros / 1_000_000, 3)
    return info


def _extract_tool_calls(llm_output: dict | None) -> list[dict]:
    """Extract tool call dicts from an LLM output payload.

    ``llm_output`` is the raw dict from a ``ScopeEndEvent.output`` on a
    ``scope_type == "llm"`` event. Handles the flat ``output.tool_calls`` shape;
    OpenAI-style ``output.choices[0].message.tool_calls`` unwrapping is left to
    a future enhancement (see the converter companion doc).
    """
    if not llm_output:
        return []
    raw_calls = llm_output.get("tool_calls") or []
    result = []
    for tc in raw_calls:
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        result.append(
            {
                "tool_call_id": tc["id"],
                "function_name": tc["name"],
                "arguments": args,
            }
        )
    return result


def _unwrap_llm_messages(input_obj: dict | None) -> list[dict]:
    """Extract messages from an LLM ScopeStart input payload.

    Handles both envelope format (``{"content": {"messages": [...]}}``)
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
    """Return sort key for stable tool ordering by LLM declaration order."""
    try:
        return order_list.index(sort_id)
    except ValueError:
        return len(order_list)


# ---------------------------------------------------------------------------
# Core accumulator
# ---------------------------------------------------------------------------


def _events_to_step_dicts(events: list[Event]) -> list[dict]:
    """Convert typed ATOF events to step dicts using the accumulator pattern.

    Dispatch (spec §3 + atof-to-atif-converter.md §3):
    - ScopeStart (scope_type=='llm')  → user step (messages from input)
    - ScopeEnd   (scope_type=='llm')  → agent step (tool_calls from output)
    - ScopeEnd   (scope_type=='tool') → buffered observation, flushed on next LLM turn
    - ScopeStart/ScopeEnd (other scope_types) → skip (structural)
    - Mark (with data != null) → system step
    """
    # Sort by normalized microsecond timestamp (spec §6.1). Polymorphic str|int
    # timestamps would otherwise raise TypeError on mixed compare.
    sorted_events = sorted(events, key=lambda e: e.ts_micros)

    # Pre-pass: build uuid → name + uuid → start_ts_micros maps
    name_map: dict[str, str] = {}
    start_ts_map: dict[str, int] = {}

    for event in sorted_events:
        if event.uuid and event.name:
            name_map[event.uuid] = event.name
        if isinstance(event, ScopeStartEvent):
            start_ts_map[event.uuid] = event.ts_micros

    # Accumulator state
    step_dicts: list[dict] = []
    pending_observations: list[dict] = []
    pending_obs_timestamp: str | int | None = None
    last_tool_call_order: list[str] = []
    current_agent_step_idx: int | None = None
    pending_tool_ancestry: list[dict] = []
    pending_tool_invocations: list[dict] = []

    def flush_observations() -> None:
        nonlocal pending_observations, pending_obs_timestamp
        if not pending_observations:
            return
        sanitized = [{"content": obs["content"]} for obs in pending_observations]
        step_dicts.append(
            {
                "source": "system",
                "message": "",
                "timestamp": pending_obs_timestamp,
                "observation": {"results": sanitized},
            }
        )
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
        if isinstance(event, ScopeStartEvent) and event.scope_type == "llm":
            flush_observations()
            finalize_agent_extra()

            input_obj = event.input if isinstance(event.input, dict) else {}
            messages = _unwrap_llm_messages(input_obj)
            ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            message_str = json.dumps(messages, separators=(",", ":")) if messages else ""

            step_dicts.append(
                {
                    "source": "user",
                    "message": message_str,
                    "timestamp": event.timestamp,
                    "extra": {"ancestry": ancestry},
                }
            )

        elif isinstance(event, ScopeEndEvent) and event.scope_type == "llm":
            flush_observations()

            raw_output = event.output if isinstance(event.output, dict) else {}
            tool_call_dicts = _extract_tool_calls(raw_output)

            last_tool_call_order.clear()
            for tc in tool_call_dicts:
                last_tool_call_order.append(tc["tool_call_id"])

            ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, event.uuid)

            extra: dict = {"ancestry": ancestry, "invocation": invocation}

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

        elif isinstance(event, ScopeEndEvent) and event.scope_type == "tool":
            # tool_call_id is read directly from the typed field on the event (spec §1.2).
            tool_call_id = event.tool_call_id

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

            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, tool_call_id or event.uuid)
            pending_tool_invocations.append(invocation)

        elif isinstance(event, MarkEvent) and event.data is not None:
            flush_observations()
            finalize_agent_extra()
            step_dicts.append(
                {
                    "source": "system",
                    "message": json.dumps(event.data, separators=(",", ":"))
                    if isinstance(event.data, dict)
                    else str(event.data),
                    "timestamp": event.timestamp,
                }
            )

        else:
            logger.debug(
                "Skipping %s (scope_type=%s) event: %s",
                event.kind,
                getattr(event, "scope_type", None),
                event.name,
            )

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
    """Convert a list of ATOF events to an ATIF Trajectory."""
    step_dicts = _events_to_step_dicts(events)

    # Extract agent info from events. scope_type is a canonical closed enum
    # (spec §4); dispatch on the string literal.
    agent_name = "unknown"
    model_name: str | None = None
    session_id = "atof-session"

    for event in events:
        if isinstance(event, ScopeStartEvent) and event.scope_type == "agent":
            agent_name = event.name
            break

    for event in events:
        if isinstance(event, ScopeEndEvent) and event.scope_type == "llm":
            # model_name lives directly on the typed field (spec §1.2).
            if event.model_name:
                model_name = event.model_name
                break
            model_name = event.name  # fallback to span name

    return Trajectory(
        session_id=session_id,
        agent=Agent(name=agent_name, version="1.0.0", model_name=model_name),
        steps=[Step(**sd) for sd in step_dicts],
    )


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Trajectory:
    """Read an ATOF JSON-Lines file and convert to an ATIF Trajectory."""
    events = read_jsonl(input_path)
    trajectory = convert(events)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traj_dict = trajectory.model_dump(exclude_none=True, mode="json")
        output_path.write_text(json.dumps(traj_dict, indent=2) + "\n")

    return trajectory
