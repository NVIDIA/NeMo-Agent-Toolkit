# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF-to-ATIF converter.

Converts a list of ATOF events (JSON-Lines wire format from agent runtime
subscriber callbacks) into an ATIF Trajectory using NAT's native models.

Implements the accumulator state machine described in
``atof-event-format.md`` Section 7. Uses ``nat.atof`` for input models
and ``nat.atif`` for output models.

Event model: 4 event types (ScopeStart/End + Mark + StreamHeaderEvent) per
spec v0.2. Dispatch keys on ``(kind, scope_type: str, profile.$schema)``
tuples — the closed scope-type enum and the v0.1 per-scope-type typed
profile classes were removed in phase 8 (see ``atof-event-format.md``
§3.1, §4, §6). StreamHeaderEvents are consumed by a pre-pass to build
the stream's schema registry and default profile mode; the main dispatch
loop skips them.

Known limitation (WR-03 from Phase 8 code review): The name-based
``last_tool_call_map`` fallback used when a tool ``ScopeEndEvent`` carries
no ``default/tool.v1`` profile (or no ``tool_call_id`` field) keys tool
observations by ``function_name``. If two tool invocations share the same
function name within a single LLM turn (e.g., two ``calculator__add``
calls), the second ``tool_call_id`` overwrites the first in the map and
both observations pair with the SAME (later) id. This is masked for
streams that carry ``default/tool.v1`` profiles with explicit
``tool_call_id`` (the primary v0.2 reference path). Opaque/vendor-profile
streams MUST NOT repeat tool names within a single LLM turn, or consumers
MUST supply an explicit ``tool_call_id`` via a ``default/tool.v1``-shaped
profile. A FIFO queue keyed on ``function_name`` is a future extension if
opaque-profile multi-call-same-tool scenarios become common.
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
from nat.atof.events import StreamHeaderEvent
from nat.atof.io import read_jsonl
from nat.atof.profile_contract import ProfileContract  # noqa: F401  (type reference in docstrings)
from nat.atof.profiles import DefaultLlmV1  # noqa: F401  (reference profile; surface available to callers)
from nat.atof.profiles import DefaultToolV1  # noqa: F401  (reference profile; surface available to callers)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema_id_string(schema_id: str | dict) -> str | None:
    """Extract a canonical string ID from either string-ID or inline-schema form.

    Per ATOF v0.2 §4.1, ``$schema`` on a profile is EITHER a string ID
    (e.g., ``'default/llm.v1'``) OR an inline JSON Schema dict (which MUST
    contain ``$id``). This helper normalizes both forms to the string ID.
    Returns ``None`` for any other shape (defensive; should not occur for
    well-formed profiles).
    """
    if isinstance(schema_id, str):
        return schema_id
    if isinstance(schema_id, dict):
        return schema_id.get("$id")
    return None


def _build_ancestry(uuid: str, name: str, parent_uuid: str | None, name_map: dict[str, str]) -> dict:
    """Build ancestry dict from event identity and name lookup map."""
    return {
        "function_id": uuid,
        "function_name": name,
        "parent_id": parent_uuid or "",
        "parent_name": name_map.get(parent_uuid or "", "unknown"),
    }


def _build_invocation_info(start_micros: int | None, end_micros: int | None, invocation_id: str) -> dict:
    """Build invocation info dict from microsecond timestamps and ID.

    Accepts normalized ``ts_micros`` integers (spec §5.1) and emits the
    ATIF-expected seconds-as-float shape for ``start_timestamp`` /
    ``end_timestamp`` at millisecond precision.
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

    ``llm_output`` is the raw dict from a ``ScopeEndEvent.output`` on an
    ``scope_type="llm"`` event. The caller has already coerced non-dict
    outputs to ``{}``. Returns a list of ATIF-shaped tool-call dicts
    (``tool_call_id``, ``function_name``, ``arguments``).
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
    """Extract messages from LLMStart input payload.

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
    - ScopeStart (scope_type=llm) → user step (messages from input)
    - ScopeEnd   (scope_type=llm) → agent step (tool_calls from output)
    - ScopeEnd   (scope_type=tool) → buffered observation, flushed on next LLMStart/Mark/end-of-stream
    - Other ScopeStart/ScopeEnd (agent/function/custom/etc.) → skip (structural events)
    - Mark (with data) → system step
    - StreamHeaderEvent → consumed by pre-pass (schema registry + default mode); skipped in main loop
    """
    # Sort by normalized microsecond timestamp (spec §5.1, D-11). Polymorphic
    # str | int timestamps would otherwise raise TypeError on mixed compare.
    sorted_events = sorted(events, key=lambda e: e.ts_micros)

    # Pre-pass: build uuid → name and uuid → start_ts_micros maps
    name_map: dict[str, str] = {}
    start_ts_map: dict[str, int] = {}

    for event in sorted_events:
        if event.uuid and event.name:
            name_map[event.uuid] = event.name
        if isinstance(event, ScopeStartEvent):
            start_ts_map[event.uuid] = event.ts_micros

    # Pre-pass: build schema registry from all StreamHeaderEvents.
    # Per D-08: multiple headers allowed; later schema defs supersede earlier.
    # Per D-09: absent any header → effective_mode="opaque".
    # These values are reserved for optional consumer-side validation — the
    # current converter does not validate, but the registry and mode are built
    # so future extensions (e.g., per-event profile validation against the
    # declared `$schema`) don't need another pre-pass.
    effective_mode: str = "opaque"
    schema_registry: dict[str, dict] = {}
    for event in sorted_events:
        if isinstance(event, StreamHeaderEvent):
            effective_mode = event.profile_mode_default
            schema_registry.update(event.schemas)
    logger.debug("ATOF stream header: effective_mode=%s, %d schemas registered", effective_mode, len(schema_registry))

    # Accumulator state
    step_dicts: list[dict] = []
    pending_observations: list[dict] = []
    pending_obs_timestamp: str | int | None = None
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
        if isinstance(event, StreamHeaderEvent):
            # Already consumed by the schema-registry pre-pass; skip main dispatch.
            continue

        if isinstance(event, ScopeStartEvent) and event.scope_type == "llm":
            # v0.2 dispatch: match on open-vocabulary scope_type string (spec §3.1, D-10).
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
            # v0.2 dispatch: model_name is read from the reference-profile vendor
            # field when the profile declares `$schema == "default/llm.v1"`; any
            # other schema falls through with no typed extraction (D-16 opaque
            # passthrough — the profile itself is still preserved on the event).
            flush_observations()

            raw_output = event.output if isinstance(event.output, dict) else {}
            tool_call_dicts = _extract_tool_calls(raw_output)

            # WR-03 (Phase 8 review): name-keyed map is a fallback for opaque /
            # vendor-profile tool events that lack an explicit `tool_call_id`.
            # Repeated tool names within a single LLM turn silently overwrite the
            # earlier id — the limitation is documented in the module docstring.
            # The primary v0.2 reference path carries `default/tool.v1` with an
            # explicit `tool_call_id`, which is read at the tool-ScopeEnd branch
            # below BEFORE consulting this fallback map.
            last_tool_call_map.clear()
            last_tool_call_order.clear()
            for tc in tool_call_dicts:
                last_tool_call_map[tc["function_name"]] = tc["tool_call_id"]
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
            # v0.2 dispatch: tool_call_id is read from the reference-profile vendor
            # field when the profile declares `$schema == "default/tool.v1"`. Any
            # other schema falls through to the name-based lookup (last_tool_call_map).
            # Vendor fields on wire-deserialized profiles live in `model_extra` on the
            # base `ProfileContract` class — attribute access would silently return
            # `None` (Pitfall 7), so we use `model_dump(by_alias=True).get(...)`.
            tool_call_id: str | None = None
            if event.profile is not None:
                schema_id = _schema_id_string(event.profile.schema_id)
                if schema_id == "default/tool.v1":
                    # Vendor-field extraction via wire-alias dump (Pitfall 7): wire-deserialized
                    # profiles are base `ProfileContract` instances; `tool_call_id` lives in
                    # `model_extra`. Attribute access would silently return `None`.
                    raw = event.profile.model_dump(by_alias=True).get("tool_call_id")
                    if isinstance(raw, str):
                        tool_call_id = raw
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
                "Skipping %s (scope_type=%s) event: %s", event.kind, getattr(event, "scope_type", None), event.name
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
    """Convert a list of ATOF events to an ATIF Trajectory.

    Args:
        events: List of typed ATOF Event objects.

    Returns:
        A validated ATIF Trajectory.
    """
    step_dicts = _events_to_step_dicts(events)

    # Extract agent info from events. scope_type is an open-vocabulary string
    # in v0.2 (D-10); compare against the reference convention `"agent"` /
    # `"llm"` rather than the removed ScopeType enum.
    agent_name = "unknown"
    model_name: str | None = None
    session_id = "atof-session"

    for event in events:
        if isinstance(event, ScopeStartEvent) and event.scope_type == "agent":
            agent_name = event.name
            break

    for event in events:
        if isinstance(event, ScopeEndEvent) and event.scope_type == "llm" and event.profile is not None:
            # Reference-profile vendor-field extraction (D-14): only pull model_name
            # when the profile declares `$schema == "default/llm.v1"`. Other schemas
            # fall through to the event name fallback (preserves v0.1 behavior).
            schema_id = _schema_id_string(event.profile.schema_id)
            if schema_id == "default/llm.v1":
                # Vendor-field extraction via wire-alias dump (Pitfall 7): see tool_call_id
                # extraction above for the base-class / model_extra rationale.
                raw_model_name = event.profile.model_dump(by_alias=True).get("model_name")
                model_name = raw_model_name if isinstance(raw_model_name, str) else event.name
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
