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


"""ATOF-to-ATIF converter.

Converts a list of ATOF events (JSON-Lines wire format from agent runtime
subscriber callbacks) into an ATIF Trajectory using NAT's native models.

Event model: 2 event kinds (``ScopeEvent`` / ``MarkEvent``) per ATOF spec
v0.1. Dispatch keys on ``(kind, scope_category, category)``. Category-specific
typed fields live inside the ``category_profile`` sub-object (spec §4.4) —
``model_name`` for ``llm``, ``tool_call_id`` for ``tool``.

Output conforms to ATIF v1.7. See the conversion rules in
``atif-alignment/docs/atof-to-atif-mapping.md``; rule identifiers (R1-R12)
referenced inline map to that document.

Producer-specific payload parsing is delegated to pluggable extractors
(:mod:`nat.atof.extractors`) keyed on the event's declared ``data_schema``.
Events without a matching registered extractor fall back to built-in
OpenAI-chat-completions / generic extractors. Two fail-fast guardrails
catch producers that would otherwise silently lose content:

- :class:`DataSchemaViolationError` — when the producer declares a
  ``data_schema`` registered in :mod:`nat.atof.schemas` and ``event.data``
  fails JSON-Schema validation against it. Fires in the pre-pass.
- :class:`ShapeMismatchError` — when ``event.data`` is non-empty but the
  resolved extractor yields nothing usable (payload would drop).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema

from nat.atif.agent import Agent
from nat.atif.function_ancestry import FunctionAncestry
from nat.atif.step import Step
from nat.atif.tool_call import ToolCall
from nat.atif.trajectory import Trajectory
from nat.atof.events import Event
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEvent
from nat.atof.extractors import resolve_llm_extractor
from nat.atof.extractors import resolve_mark_extractor
from nat.atof.extractors import resolve_tool_extractor
from nat.atof.io import read_jsonl
from nat.atof.schemas import lookup_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ShapeMismatchError(ValueError):
    """Raised when an event's non-empty ``data`` produced empty extraction.

    The resolved :class:`~nat.atof.extractors.LlmPayloadExtractor` for an
    event's ``data_schema`` could not pull any usable content out of a
    non-empty payload. The would-be-emitted content is silently dropped —
    this exception surfaces that case as a hard failure so callers can
    either (a) fix the producer to emit the expected shape, (b) declare a
    matching ``data_schema`` and register a profile-specific extractor
    via :func:`~nat.atof.extractors.register_llm_extractor`, or (c) wrap
    the call and handle the drop explicitly.

    Attributes:
        kind: ``"llm_input"`` or ``"llm_output"`` — which extraction missed.
        uuid: UUID of the offending event.
        data_schema: The producer-declared ``data_schema``, if any.
        data_keys: Sorted top-level keys observed in ``data``.
    """

    def __init__(
        self,
        *,
        kind: str,
        uuid: str,
        data_schema: dict[str, Any] | None,
        data_keys: list[str],
    ):
        self.kind = kind
        self.uuid = uuid
        self.data_schema = data_schema
        self.data_keys = data_keys
        super().__init__(f"ATOF→ATIF would drop data on {kind} event (uuid={uuid}): "
                         "the payload did not match the converter's extraction assumptions. "
                         f"data_schema={data_schema}, data_keys={data_keys}")


class DataSchemaViolationError(ValueError):
    """Raised when an event declares a registered ``data_schema`` but its
    ``data`` fails JSON-Schema validation against it.

    Producers declaring a schema enter a contract: their payload MUST
    conform. A violation here either reveals a producer bug or signals
    that the declared schema is wrong. Either way, downstream extraction
    would likely drop content, so the converter fails fast with actionable
    context — the offending event UUID, the declared schema identifier,
    the JSON-pointer path to the validation failure, and the underlying
    validator message.

    Events whose ``data_schema`` is NOT in the registry skip validation
    entirely (a ``WARNING`` is logged instead).

    Attributes:
        uuid: UUID of the offending event.
        data_schema: The producer-declared ``{name, version}`` identifier.
        path: JSON-pointer segments to the offending value.
        message: The underlying ``jsonschema`` validator message.
    """

    def __init__(
        self,
        *,
        uuid: str,
        data_schema: dict[str, Any],
        path: list[Any],
        message: str,
    ):
        self.uuid = uuid
        self.data_schema = data_schema
        self.path = path
        self.message = message
        super().__init__(f"ATOF event (uuid={uuid}) data violates its declared "
                         f"data_schema {data_schema}: {message} "
                         f"(at {path or '<root>'})")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_event_data_schema(event: Event) -> None:
    """Validate ``event.data`` against its declared, registered ``data_schema``.

    - Events without a ``data_schema`` pass through untouched (the schema
      field is optional per spec §2).
    - Events with a ``data_schema`` not in :data:`nat.atof.schemas.SCHEMA_REGISTRY`
      emit a ``WARNING`` and pass through; producers can register custom
      schemas via :func:`nat.atof.schemas.register_schema`.
    - Events with a registered schema raise :class:`DataSchemaViolationError`
      on validation failure.
    """
    ds = event.data_schema
    if not ds:
        return
    name = ds.get("name") if isinstance(ds, dict) else None
    version = ds.get("version") if isinstance(ds, dict) else None
    if not isinstance(name, str) or not isinstance(version, str):
        return
    schema = lookup_schema(name, version)
    if schema is None:
        logger.warning(
            "ATOF event %s declares unregistered data_schema %s@%s; "
            "validation skipped. Register the schema via "
            "nat.atof.schemas.register_schema() to enable validation.",
            event.uuid,
            name,
            version,
        )
        return
    try:
        jsonschema.validate(instance=event.data, schema=schema)
    except jsonschema.ValidationError as exc:
        raise DataSchemaViolationError(
            uuid=event.uuid,
            data_schema=ds,
            path=list(exc.absolute_path),
            message=exc.message,
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ancestry(uuid: str, name: str, parent_uuid: str | None, name_map: dict[str, str]) -> dict:
    """Build a v1.7 FunctionAncestry dict (parent_id/parent_name null at root)."""
    parent_name = name_map.get(parent_uuid) if parent_uuid else None
    return {
        "function_id": uuid,
        "function_name": name,
        "parent_id": parent_uuid,
        "parent_name": parent_name,
    }


def _build_invocation_info(start_micros: int | None, end_micros: int | None, invocation_id: str) -> dict:
    """Build producer-scoped invocation info for step.extra (not part of ATIF v1.7 core)."""
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


def _is_scope_start(event: Event) -> bool:
    return isinstance(event, ScopeEvent) and event.scope_category == "start"


def _is_scope_end(event: Event) -> bool:
    return isinstance(event, ScopeEvent) and event.scope_category == "end"


def _build_category_map(events: list[Event]) -> dict[str, str]:
    """UUID → category lookup from scope-start events."""
    cat_map: dict[str, str] = {}
    for e in events:
        if _is_scope_start(e) and isinstance(e, ScopeEvent) and e.category:
            cat_map[e.uuid] = e.category
    return cat_map


def _build_parent_map(events: list[Event]) -> dict[str, str | None]:
    """UUID → parent_uuid for all unique UUIDs in the stream."""
    parent_map: dict[str, str | None] = {}
    for e in events:
        if e.uuid and e.uuid not in parent_map:
            parent_map[e.uuid] = e.parent_uuid
    return parent_map


def _find_subagent_roots(events: list[Event], category_map: dict[str, str]) -> list[ScopeEvent]:
    """Find agent scope-starts whose parent is a dispatcher scope (R7).

    A dispatcher scope is a ``tool`` scope (regular delegation) or a
    ``context`` scope (R10 context-management subagent, e.g. a compaction
    subagent that summarizes prior turns).
    """
    roots: list[ScopeEvent] = []
    for e in events:
        if (_is_scope_start(e) and isinstance(e, ScopeEvent) and e.category == "agent" and e.parent_uuid is not None
                and category_map.get(e.parent_uuid) in ("tool", "context")):
            roots.append(e)
    return roots


def _collect_descendants(root_uuid: str, events: list[Event], parent_map: dict[str, str | None]) -> list[Event]:
    """Events whose ancestry chain reaches root_uuid (inclusive of events with uuid == root_uuid).

    ``events`` preserves the caller's order; the returned list preserves it too.
    """
    result: list[Event] = []
    for e in events:
        u = e.uuid
        depth = 0
        while u is not None and depth < 64:  # guard against cycles
            if u == root_uuid:
                result.append(e)
                break
            u = parent_map.get(u)
            depth += 1
    return result


# ---------------------------------------------------------------------------
# Core accumulator (ATIF v1.7 emission)
# ---------------------------------------------------------------------------


def _events_to_step_dicts(
    events: list[Event],
    subagent_ref_by_tc_id: dict[str, dict] | None = None,
    subagent_ref_by_context_uuid: dict[str, dict] | None = None,
) -> list[dict]:
    """Convert typed ATOF events to ATIF v1.7 step dicts.

    ``subagent_ref_by_tc_id`` maps a ``tool_call_id`` to a
    ``SubagentTrajectoryRef``-shaped dict (R7 tool-wraps-agent).

    ``subagent_ref_by_context_uuid`` maps a ``context``-scope UUID to a
    ``SubagentTrajectoryRef``-shaped dict (R10 context-wrapped subagent,
    e.g. a compaction subagent). Either map MAY be empty.

    Raises:
        DataSchemaViolationError: if an event declares a registered
            ``data_schema`` and its ``data`` fails validation.
        ShapeMismatchError: if an ``llm`` scope event's non-empty ``data``
            yields no extractable content (would drop payload silently).
    """
    subagent_ref_by_tc_id = subagent_ref_by_tc_id or {}
    subagent_ref_by_context_uuid = subagent_ref_by_context_uuid or {}

    sorted_events = sorted(events, key=lambda e: e.ts_micros)

    # Pre-pass
    name_map: dict[str, str] = {}
    start_ts_map: dict[str, int] = {}
    tool_start_args_by_tc_id: dict[str, dict] = {}
    for event in sorted_events:
        _validate_event_data_schema(event)

        if event.uuid and event.name:
            name_map[event.uuid] = event.name
        if _is_scope_start(event):
            start_ts_map[event.uuid] = event.ts_micros
            # Cache tool scope-start arguments for R13 (no-LLM orchestration)
            # synthesis — function scope-ends need tool_call args from scope-starts.
            if isinstance(event, ScopeEvent) and event.category == "tool":
                tc_id = (event.category_profile or {}).get("tool_call_id")
                if tc_id:
                    tool_start_args_by_tc_id[tc_id] = (event.data if isinstance(event.data, dict) else {})

    # Streaming state
    step_dicts: list[dict] = []
    pending_observations: list[dict] = []
    pending_obs_timestamp: str | int | None = None
    pending_tool_ancestry_by_id: dict[str, dict] = {}
    pending_tool_invocations: list[dict] = []
    current_agent_step_idx: int | None = None
    # Per (parent_uuid, role) → set of already-emitted content strings.
    # Used for R2/R3 (user turns) and extended role=system handling — lets
    # each NEW role=user / role=system message in an LLM's input seed a
    # new step, which naturally models multi-turn conversations.
    seen_input_messages: dict[tuple[str | None, str], set[str]] = {}

    def flush_observations() -> None:
        """Attach buffered observations to the preceding agent step (R4 drain)."""
        nonlocal pending_observations, pending_obs_timestamp
        nonlocal pending_tool_ancestry_by_id, pending_tool_invocations

        if not pending_observations and not pending_tool_ancestry_by_id:
            return

        def _build_results(obs_list: list[dict]) -> list[dict]:
            results = []
            for obs in obs_list:
                entry: dict = {"content": obs["content"]}
                if obs.get("source_call_id"):
                    entry["source_call_id"] = obs["source_call_id"]
                if obs.get("subagent_trajectory_ref"):
                    entry["subagent_trajectory_ref"] = obs["subagent_trajectory_ref"]
                results.append(entry)
            return results

        if current_agent_step_idx is not None:
            agent_step = step_dicts[current_agent_step_idx]

            if pending_observations:
                agent_step["observation"] = {"results": _build_results(pending_observations)}

            if pending_tool_ancestry_by_id and agent_step.get("tool_calls"):
                for tc in agent_step["tool_calls"]:
                    anc = pending_tool_ancestry_by_id.get(tc["tool_call_id"])
                    if anc:
                        tc["tool_ancestry"] = anc

            if pending_tool_invocations:
                extra = dict(agent_step.get("extra") or {})
                extra["tool_invocations"] = pending_tool_invocations
                agent_step["extra"] = extra
        elif pending_observations:
            step_dicts.append({
                "source": "system",
                "message": "",
                "timestamp": pending_obs_timestamp,
                "observation": {
                    "results": _build_results(pending_observations)
                },
            })

        pending_observations = []
        pending_tool_ancestry_by_id = {}
        pending_tool_invocations = []
        pending_obs_timestamp = None

    # Main event loop
    for event in sorted_events:
        if _is_scope_start(event) and event.category == "llm":
            flush_observations()

            # R2/R3 (multi-turn aware): emit user/system steps for every NEW
            # role=user or role=system message in the LLM's input. A
            # continuation LLM call under the same agent where the user has
            # said nothing new emits no step; a follow-up user turn (new
            # content) emits one. System prompts surface as source=system
            # steps the first time they appear.
            data = event.data if isinstance(event.data, dict) else {}
            llm_extractor = resolve_llm_extractor(event.data_schema)
            messages = llm_extractor.extract_input_messages(data)
            if data and not messages:
                raise ShapeMismatchError(
                    kind="llm_input",
                    uuid=event.uuid,
                    data_schema=event.data_schema,
                    data_keys=sorted(data.keys()),
                )
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role not in ("user", "system"):
                    continue
                # Multimodal content (ATIF v1.6+ ContentPart[]) is passed
                # through; dedup key for list content is a canonical JSON
                # representation of the list.
                if isinstance(content, str):
                    dedup_key = content
                    emit_content = content
                elif isinstance(content, list):
                    dedup_key = json.dumps(content, sort_keys=True, separators=(",", ":"))
                    emit_content = content
                else:
                    continue

                key = (event.parent_uuid, role)
                seen = seen_input_messages.setdefault(key, set())
                if dedup_key not in seen:
                    step_dicts.append({
                        "source": role,
                        "message": emit_content,
                        "timestamp": event.timestamp,
                    })
                    seen.add(dedup_key)
                    # A new user/system step breaks any active agent
                    # observation window (it's a fresh turn, not a
                    # continuation of the previous agent step).
                    current_agent_step_idx = None

        elif _is_scope_end(event) and event.category == "llm":
            flush_observations()

            raw_data = event.data if isinstance(event.data, dict) else {}
            llm_extractor = resolve_llm_extractor(event.data_schema)
            tool_call_dicts = llm_extractor.extract_tool_calls(raw_data)
            agent_msg = llm_extractor.extract_output_text(raw_data)
            # A payload that yields NEITHER assistant content NOR tool_calls
            # would drop the producer's response entirely. A payload with
            # only tool_calls (no content) or only content (no tool_calls)
            # is legitimate and not an error.
            if raw_data and not agent_msg and not tool_call_dicts:
                raise ShapeMismatchError(
                    kind="llm_output",
                    uuid=event.uuid,
                    data_schema=event.data_schema,
                    data_keys=sorted(raw_data.keys()),
                )
            function_ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)

            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, event.uuid)

            extra_fields: dict = {"invocation": invocation}
            # Producer extension: preserve data_schema declared by the producer
            # on the LLM scope-end (consumer may want to validate data shape).
            if event.data_schema:
                extra_fields["data_schema"] = event.data_schema

            step_dict: dict = {
                "source": "agent",
                "message": agent_msg,
                "timestamp": event.timestamp,
                "function_ancestry": function_ancestry,
                "llm_call_count": 1,
                "extra": extra_fields,
            }
            if tool_call_dicts:
                step_dict["tool_calls"] = tool_call_dicts

            step_dicts.append(step_dict)
            current_agent_step_idx = len(step_dicts) - 1

        elif _is_scope_end(event) and event.category == "tool":
            tool_call_id = (event.category_profile or {}).get("tool_call_id")
            if pending_obs_timestamp is None:
                pending_obs_timestamp = event.timestamp

            tool_extractor = resolve_tool_extractor(event.data_schema)
            content = tool_extractor.extract_tool_result(event.data)
            obs_entry: dict = {"source_call_id": tool_call_id, "content": content}
            if tool_call_id and tool_call_id in subagent_ref_by_tc_id:
                obs_entry["subagent_trajectory_ref"] = [subagent_ref_by_tc_id[tool_call_id]]
            pending_observations.append(obs_entry)

            if tool_call_id:
                pending_tool_ancestry_by_id[tool_call_id] = _build_ancestry(event.uuid,
                                                                            event.name,
                                                                            event.parent_uuid,
                                                                            name_map)

            start_micros = start_ts_map.get(event.uuid)
            pending_tool_invocations.append(
                _build_invocation_info(start_micros, event.ts_micros, tool_call_id or event.uuid))

        elif isinstance(event, MarkEvent) and event.data is not None:
            flush_observations()
            current_agent_step_idx = None

            data = event.data
            mark_extractor = resolve_mark_extractor(event.data_schema)
            role_and_content = mark_extractor.extract_role_and_content(data)
            if role_and_content is not None:
                # R9 extension: a mark whose payload names an ATIF step source
                # emits that step directly. This lets no-LLM producers surface
                # user turns and clean system messages without an LLM scope.
                source, content = role_and_content
                step_dict = {
                    "source": source,
                    "message": content if isinstance(content, str) else json.dumps(content, separators=(",", ":")),
                    "timestamp": event.timestamp,
                }
                # Track user/system content so subsequent LLM input scanners don't
                # re-emit it (same dedup path as R2/R3).
                if source in ("user", "system") and isinstance(content, str):
                    seen_input_messages.setdefault((event.parent_uuid, source), set()).add(content)
                step_dicts.append(step_dict)
            else:
                step_dicts.append({
                    "source": "system",
                    "message": json.dumps(data, separators=(",", ":")) if isinstance(data, dict) else str(data),
                    "timestamp": event.timestamp,
                })

        elif _is_scope_end(event) and event.category == "context":
            # R10: context-window transformation boundary. Emit a system step
            # with extra.context_management populated from category_profile.
            # If the context scope wrapped a subagent (e.g. compaction agent),
            # attach subagent_trajectory_ref to the observation.
            flush_observations()
            current_agent_step_idx = None

            profile = event.category_profile or {}
            data = event.data if isinstance(event.data, dict) else None

            # Unwrap single-key {summary|result: X} to primitive content (R5-style).
            content: str | None = None
            if isinstance(data, dict) and data:
                if len(data) == 1 and next(iter(data)) in ("summary", "result"):
                    val = next(iter(data.values()))
                    content = val if isinstance(val, str) else json.dumps(val, separators=(",", ":"))
                else:
                    content = json.dumps(data, separators=(",", ":"))

            step_extra: dict = {
                "context_management": {
                    "type": profile.get("type"),
                    "boundary": profile.get("boundary"),
                }
            }
            if event.data_schema:
                step_extra["data_schema"] = event.data_schema

            step_dict: dict = {
                "source": "system",
                "message": event.name or "context_management",
                "timestamp": event.timestamp,
                "extra": step_extra,
            }

            subagent_ref = subagent_ref_by_context_uuid.get(event.uuid)
            if content is not None or subagent_ref is not None:
                entry: dict = {}
                if content is not None:
                    entry["content"] = content
                if subagent_ref is not None:
                    entry["subagent_trajectory_ref"] = [subagent_ref]
                step_dict["observation"] = {"results": [entry]}

            step_dicts.append(step_dict)

            # R10 boundary-replace dedup: for boundary="replace", the compaction
            # summary REPLACES prior context — producers will typically include
            # the summary as a role="system" message on the next LLM's input.
            # Mark it as already-seen so the multi-turn input scanner doesn't
            # re-emit it as a standalone system step.
            if (content is not None and profile.get("boundary") == "replace" and event.parent_uuid is not None):
                seen_input_messages.setdefault((event.parent_uuid, "system"), set()).add(content)

        elif _is_scope_end(event) and event.category == "function" and pending_observations:
            # R13 (v1.7-alignment-proposal): a ``function`` scope that contained
            # tool scope-ends is a deterministic dispatcher — no LLM was
            # consulted, but tool_calls were issued. Emit an agent step with
            # llm_call_count=0 and synthesize tool_calls from the buffered
            # tool scope data (R6 flattening still applies for nested tools).
            synthetic_tcs: list[dict] = []
            for obs in pending_observations:
                tc_id = obs.get("source_call_id")
                if not tc_id:
                    continue
                anc = pending_tool_ancestry_by_id.get(tc_id, {})
                synthetic_tcs.append({
                    "tool_call_id": tc_id,
                    "function_name": anc.get("function_name", "unknown"),
                    "arguments": tool_start_args_by_tc_id.get(tc_id, {}),
                })

            function_ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, event.uuid)

            r13_extra: dict = {"invocation": invocation}
            if event.data_schema:
                r13_extra["data_schema"] = event.data_schema
            step_dict = {
                "source": "agent",
                "message": "",
                "timestamp": event.timestamp,
                "function_ancestry": function_ancestry,
                "llm_call_count": 0,
                "tool_calls": synthetic_tcs,
                "extra": r13_extra,
            }
            step_dicts.append(step_dict)
            current_agent_step_idx = len(step_dicts) - 1
            # flush_observations will now drain pending obs + tool_ancestry
            # into this newly-emitted orchestrator step.
            flush_observations()

        elif _is_scope_end(event) and event.category not in ("llm", "tool", "agent", "context"):
            flush_observations()
            current_agent_step_idx = None

            data = event.data
            if isinstance(data, dict):
                message = json.dumps(data, separators=(",", ":"))
            elif isinstance(data, str):
                message = data
            elif data is not None:
                message = str(data)
            else:
                message = ""

            function_ancestry = _build_ancestry(event.uuid, event.name, event.parent_uuid, name_map)
            start_micros = start_ts_map.get(event.uuid)
            invocation = _build_invocation_info(start_micros, event.ts_micros, event.uuid)

            r8_extra: dict = {"invocation": invocation}
            if event.data_schema:
                r8_extra["data_schema"] = event.data_schema
            step_dicts.append({
                "source": "system",
                "message": message,
                "timestamp": event.timestamp,
                "function_ancestry": function_ancestry,
                "extra": r8_extra,
            })

        else:
            logger.debug(
                "Skipping %s (scope_category=%s, category=%s) event: %s",
                event.kind,
                getattr(event, "scope_category", None),
                getattr(event, "category", None),
                event.name,
            )

    flush_observations()

    for i, step in enumerate(step_dicts):
        step["step_id"] = i + 1

    return step_dicts


def _materialize_steps(step_dicts: list[dict]) -> list[Step]:
    """Build validated Step instances from raw step dicts."""
    steps = []
    for sd in step_dicts:
        tool_calls = None
        if sd.get("tool_calls"):
            tool_calls = []
            for tc in sd["tool_calls"]:
                anc = tc.pop("tool_ancestry", None)
                tc_kwargs = dict(tc)
                if anc is not None:
                    tc_kwargs["tool_ancestry"] = FunctionAncestry(**anc)
                tool_calls.append(ToolCall(**tc_kwargs))

        step_kwargs = {k: v for k, v in sd.items() if k != "tool_calls"}
        if "function_ancestry" in step_kwargs and step_kwargs["function_ancestry"] is not None:
            step_kwargs["function_ancestry"] = FunctionAncestry(**step_kwargs["function_ancestry"])
        if tool_calls is not None:
            step_kwargs["tool_calls"] = tool_calls
        steps.append(Step(**step_kwargs))
    return steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert(events: list[Event]) -> Trajectory:
    """Convert a list of ATOF events to an ATIF v1.7 Trajectory.

    Raises:
        DataSchemaViolationError: if an event declares a registered
            ``data_schema`` (see :mod:`nat.atof.schemas`) and its ``data``
            fails JSON-Schema validation.
        ShapeMismatchError: if an ``llm`` scope event carries non-empty
            ``data`` that the reference extractors cannot parse. Silently
            dropping such a payload would lose producer content, so the
            converter fails fast instead.
    """
    return _convert_impl(events, explicit_root_uuid=None)


def _convert_impl(events: list[Event], explicit_root_uuid: str | None) -> Trajectory:
    """Internal converter supporting recursion on subagent sub-streams.

    When ``explicit_root_uuid`` is provided (recursive call), the root agent
    metadata is taken from the event with ``uuid == explicit_root_uuid``
    rather than by searching for ``parent_uuid is None``.
    """
    category_map = _build_category_map(events)
    parent_map = _build_parent_map(events)

    # R7: detect subagent roots and partition out their sub-streams
    subagent_roots = _find_subagent_roots(events, category_map)

    excluded_ids: set[int] = set()
    subagent_trajectories: list[Trajectory] = []
    subagent_ref_by_tc_id: dict[str, dict] = {}
    subagent_ref_by_context_uuid: dict[str, dict] = {}

    for root in subagent_roots:
        descendants = _collect_descendants(root.uuid, events, parent_map)
        for e in descendants:
            excluded_ids.add(id(e))

        child_trajectory = _convert_impl(descendants, explicit_root_uuid=root.uuid)
        subagent_trajectories.append(child_trajectory)

        # Correlate the child trajectory with its wrapping dispatcher scope so
        # the main pass can attach subagent_trajectory_ref to the right
        # observation. ``tool`` wrappers correlate via tool_call_id (R7);
        # ``context`` wrappers correlate via the wrapping scope's UUID (R10).
        wrapping_uuid = root.parent_uuid
        wrapping_category = None
        wrapping_tc_id = None
        if wrapping_uuid is not None:
            for e in events:
                if (_is_scope_start(e) and isinstance(e, ScopeEvent) and e.uuid == wrapping_uuid):
                    wrapping_category = e.category
                    if e.category == "tool":
                        wrapping_tc_id = (e.category_profile or {}).get("tool_call_id")
                    break

        ref = {"session_id": child_trajectory.session_id, "trajectory_path": None}
        if wrapping_category == "tool" and wrapping_tc_id:
            subagent_ref_by_tc_id[wrapping_tc_id] = ref
        elif wrapping_category == "context" and wrapping_uuid:
            subagent_ref_by_context_uuid[wrapping_uuid] = ref

    main_events = [e for e in events if id(e) not in excluded_ids]

    # Trajectory metadata extraction
    agent_name: str | None = None
    agent_version: str = "1.0.0"
    model_name: str | None = None
    session_id: str | None = None
    root_agent_uuid: str | None = None

    if explicit_root_uuid is not None:
        for event in events:
            if _is_scope_start(event) and event.uuid == explicit_root_uuid:
                agent_name = event.name
                root_agent_uuid = event.uuid
                if event.metadata and isinstance(event.metadata, dict):
                    v = event.metadata.get("version")
                    if isinstance(v, str):
                        agent_version = v
                    s = event.metadata.get("session_id")
                    if isinstance(s, str):
                        session_id = s
                break
    else:
        # R1: outermost agent scope with parent_uuid None
        for event in main_events:
            if _is_scope_start(event) and event.category == "agent" and event.parent_uuid is None:
                agent_name = event.name
                root_agent_uuid = event.uuid
                if event.metadata and isinstance(event.metadata, dict):
                    v = event.metadata.get("version")
                    if isinstance(v, str):
                        agent_version = v
                    s = event.metadata.get("session_id")
                    if isinstance(s, str):
                        session_id = s
                break

        # Tier-1 fallback
        if agent_name is None:
            for event in main_events:
                if _is_scope_start(event) and event.parent_uuid is None:
                    agent_name = event.name
                    root_agent_uuid = event.uuid
                    break

    if agent_name is None:
        agent_name = "unknown"

    if session_id is None:
        session_id = root_agent_uuid or "atof-session"

    # Pick up model_name from the first LLM scope-end (prefer ones under the root)
    for event in main_events:
        if _is_scope_end(event) and event.category == "llm":
            profile_model = (event.category_profile or {}).get("model_name")
            if profile_model:
                model_name = profile_model
                break
            model_name = event.name

    step_dicts = _events_to_step_dicts(
        main_events,
        subagent_ref_by_tc_id=subagent_ref_by_tc_id,
        subagent_ref_by_context_uuid=subagent_ref_by_context_uuid,
    )
    steps = _materialize_steps(step_dicts)

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(name=agent_name, version=agent_version, model_name=model_name),
        steps=steps,
        subagent_trajectories=subagent_trajectories or None,
    )


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Trajectory:
    """Read an ATOF JSON-Lines file and convert to an ATIF Trajectory.

    Raises:
        ShapeMismatchError: see :func:`convert`.
    """
    events = read_jsonl(input_path)
    trajectory = convert(events)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traj_dict = trajectory.model_dump(exclude_none=True, mode="json")
        _ensure_subagent_trajectory_path_explicit(traj_dict)
        output_path.write_text(json.dumps(traj_dict, indent=2) + "\n")

    return trajectory


def _ensure_subagent_trajectory_path_explicit(obj: Any) -> None:
    """Walk a dumped ATIF trajectory dict and ensure every
    ``subagent_trajectory_ref[i]`` entry has ``trajectory_path`` explicitly
    present (null for embedded refs).

    ``model_dump(exclude_none=True)`` strips optional None-valued fields,
    which produces valid ATIF v1.7 but loses back-compat visual alignment
    with ATIF v1.6 consumers that expect the key. Keeping the field
    explicit as ``null`` is spec-allowed (the field is optional, and
    ``null`` is a valid value) and aids consumer-side inspection.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "subagent_trajectory_ref" and isinstance(v, list):
                for ref in v:
                    if isinstance(ref, dict) and "trajectory_path" not in ref:
                        ref["trajectory_path"] = None
            else:
                _ensure_subagent_trajectory_path_explicit(v)
    elif isinstance(obj, list):
        for item in obj:
            _ensure_subagent_trajectory_path_explicit(item)
