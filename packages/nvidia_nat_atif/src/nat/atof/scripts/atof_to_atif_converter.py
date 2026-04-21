# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF-to-ATIF converter.

Converts a list of ATOF events (JSON-Lines wire format from agent runtime
subscriber callbacks) into an ATIF Trajectory using NAT's native models.

Event model: 2 event kinds (``ScopeEvent`` / ``MarkEvent``) per spec v0.1.
Dispatch keys on ``(kind, scope_category, category)``. Category-specific
typed fields live inside the ``category_profile`` sub-object (spec §4.4) —
``model_name`` for ``llm``, ``tool_call_id`` for ``tool``.

See ``atof-event-format.md`` §3 (event kinds), §4 (category vocabulary),
§5 (event stream semantics) and the companion
``examples/atof_to_atif/README.md`` for the normative mapping.
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
from nat.atof.events import ScopeEvent
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


def _extract_tool_calls(llm_data: dict | None) -> list[dict]:
    """Extract tool call dicts from an LLM output payload.

    ``llm_data`` is the raw dict from a ``ScopeEvent.data`` on a scope-end
    event with ``category == "llm"``. Handles the flat ``data.tool_calls``
    shape; OpenAI-style ``data.choices[0].message.tool_calls`` unwrapping is
    left to a future enhancement (see the converter companion doc).
    """
    if not llm_data:
        return []

    # Try flat shape first (NAT-native producers, EXMP-01-style).
    raw_calls = llm_data.get("tool_calls")

    # Fall back to OpenAI Chat Completions shape (data.choices[0].message.tool_calls).
    if not raw_calls:
        try:
            raw_calls = llm_data["choices"][0]["message"].get("tool_calls", [])
        except (KeyError, IndexError, TypeError):
            raw_calls = []

    result = []
    for tc in raw_calls or []:
        # Normalize OpenAI-shaped tool calls — they wrap name/arguments under
        # an inner 'function' object: {id, type, function: {name, arguments}}.
        # Flat shape uses {id, name, arguments} directly.
        if "function" in tc and isinstance(tc["function"], dict):
            inner = tc["function"]
            tool_id = tc["id"]
            name = inner.get("name", "")
            args = inner.get("arguments", {})
        else:
            tool_id = tc["id"]
            name = tc.get("name", "")
            args = tc.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        result.append(
            {
                "tool_call_id": tool_id,
                "function_name": name,
                "arguments": args,
            }
        )
    return result


def _unwrap_llm_messages(data: dict | None) -> list[dict]:
    """Extract messages from an LLM scope-start data payload.

    Handles both envelope format (``{"content": {"messages": [...]}}``)
    and direct format (``{"messages": [...]}``).
    """
    if not data:
        return []
    content = data.get("content")
    if isinstance(content, dict):
        messages = content.get("messages", [])
        if messages:
            return messages
    messages = data.get("messages", [])
    if messages:
        return messages
    return []


def _tool_call_order_key(sort_id: str, order_list: list[str]) -> int:
    """Return sort key for stable tool ordering by LLM declaration order."""
    try:
        return order_list.index(sort_id)
    except ValueError:
        return len(order_list)


def _is_scope_start(event: Event) -> bool:
    return isinstance(event, ScopeEvent) and event.scope_category == "start"


def _is_scope_end(event: Event) -> bool:
    return isinstance(event, ScopeEvent) and event.scope_category == "end"


# ---------------------------------------------------------------------------
# Core accumulator
# ---------------------------------------------------------------------------


def _events_to_step_dicts(events: list[Event]) -> list[dict]:
    """Convert typed ATOF events to step dicts using the accumulator pattern.

    Dispatch (spec §3 + examples/atof_to_atif/README.md Conversion reference):
    - scope start (category=='llm')  → user step (messages from data)
    - scope end   (category=='llm')  → agent step (tool_calls from data)
    - scope end   (category=='tool') → buffered observation, flushed on next LLM turn
    - scope start/end (other categories) → skip (structural) or generic system step
    - mark (with data != null) → system step
    """
    # Sort by normalized microsecond timestamp (spec §5.1). Polymorphic str|int
    # timestamps would otherwise raise TypeError on mixed compare.
    sorted_events = sorted(events, key=lambda e: e.ts_micros)

    # Pre-pass: build uuid → name + uuid → start_ts_micros maps
    name_map: dict[str, str] = {}
    start_ts_map: dict[str, int] = {}

    for event in sorted_events:
        if event.uuid and event.name:
            name_map[event.uuid] = event.name
        if _is_scope_start(event):
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
        if _is_scope_start(event) and event.category == "llm":
            flush_observations()
            finalize_agent_extra()

            data = event.data if isinstance(event.data, dict) else {}
            messages = _unwrap_llm_messages(data)
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

        elif _is_scope_end(event) and event.category == "llm":
            flush_observations()

            raw_data = event.data if isinstance(event.data, dict) else {}
            tool_call_dicts = _extract_tool_calls(raw_data)

            last_tool_call_order.clear()
            for tc in tool_call_dicts:
                last_tool_call_order.append(tc["tool_call_id"])

            ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, event.uuid)

            extra: dict = {"ancestry": ancestry, "invocation": invocation}

            try:
                agent_msg = raw_data["choices"][0]["message"].get("content", "")
            except (KeyError, IndexError, TypeError):
                agent_msg = json.dumps(raw_data, separators=(",", ":")) if raw_data else ""

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

        elif _is_scope_end(event) and event.category == "tool":
            # tool_call_id lives in the category_profile sub-object (spec §4.4).
            tool_call_id = (event.category_profile or {}).get("tool_call_id")

            if pending_obs_timestamp is None:
                pending_obs_timestamp = event.timestamp

            data = event.data
            if isinstance(data, dict):
                content: str | None = json.dumps(data, separators=(",", ":"))
            elif isinstance(data, str):
                content = data
            elif data is not None:
                content = str(data)
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

        elif _is_scope_end(event) and event.category not in ("llm", "tool", "agent"):
            # Opaque / generic scope end: emit a system step with the raw data
            # as the message. Covers tier-1 ("unknown") plus v0.1 categories the
            # converter does not have specialised handling for ("function",
            # "retriever", "embedder", "reranker", "guardrail", "evaluator",
            # "custom"). "agent" scopes continue to be skipped here — they only
            # contribute to Trajectory.agent.name via convert() below.
            flush_observations()
            finalize_agent_extra()

            data = event.data
            if isinstance(data, dict):
                message = json.dumps(data, separators=(",", ":"))
            elif isinstance(data, str):
                message = data
            elif data is not None:
                message = str(data)
            else:
                message = ""

            ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, event.uuid)

            step_dicts.append(
                {
                    "source": "system",
                    "message": message,
                    "timestamp": event.timestamp,
                    "extra": {"ancestry": ancestry, "invocation": invocation},
                }
            )

        else:
            logger.debug(
                "Skipping %s (scope_category=%s, category=%s) event: %s",
                event.kind,
                getattr(event, "scope_category", None),
                getattr(event, "category", None),
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

    # Extract agent info from events. Prefer an explicit `category == "agent"`
    # scope-start (tier-2+). For streams that don't classify an agent scope
    # (tier-1 opaque pass-through), fall back to the outermost root scope-start's
    # name. Final fallback is the literal "unknown".
    agent_name: str | None = None
    model_name: str | None = None
    session_id = "atof-session"

    for event in events:
        if _is_scope_start(event) and event.category == "agent":
            agent_name = event.name
            break

    if agent_name is None:
        for event in events:
            if _is_scope_start(event) and event.parent_uuid is None:
                agent_name = event.name
                break

    if agent_name is None:
        agent_name = "unknown"

    for event in events:
        if _is_scope_end(event) and event.category == "llm":
            # model_name lives in the category_profile sub-object (spec §4.4).
            profile_model = (event.category_profile or {}).get("model_name")
            if profile_model:
                model_name = profile_model
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
