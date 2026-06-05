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
"""Bridge NeMo Relay ATOF events into the NAT intermediate-step stream.

NeMo Relay emits Agent Trajectory Observability Format (ATOF) lifecycle events.
The NVIDIA NeMo Agent Toolkit telemetry pipeline consumes ``IntermediateStep``
objects. This module maps ATOF scope and mark events to ``IntermediateStep`` so
Relay-instrumented subprocesses can be folded into the active NAT trace.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.token_usage import TokenUsageBaseModel

RelayEvent = dict[str, Any]

_CATEGORY_TO_EVENT_TYPES: dict[str, tuple[IntermediateStepType, IntermediateStepType]] = {
    "agent": (IntermediateStepType.TASK_START, IntermediateStepType.TASK_END),
    "function": (IntermediateStepType.FUNCTION_START, IntermediateStepType.FUNCTION_END),
    "llm": (IntermediateStepType.LLM_START, IntermediateStepType.LLM_END),
    "tool": (IntermediateStepType.TOOL_START, IntermediateStepType.TOOL_END),
}


def load_atof_jsonl(path: str | Path) -> list[RelayEvent]:
    """Load Relay ATOF events from a JSONL file."""

    events: list[RelayEvent] = []
    with Path(path).open(encoding="utf-8") as event_file:
        for line in event_file:
            stripped = line.strip()
            if stripped:
                try:
                    event = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(event, dict):
                    events.append(event)
    return events


def relay_events_to_intermediate_steps(events: Iterable[RelayEvent],
                                       *,
                                       root_parent_id: str = "root",
                                       function_ancestry: InvocationNode | None = None) -> list[IntermediateStep]:
    """Convert Relay ATOF events to NAT intermediate steps.

    Parameters
    ----------
    events:
        Relay ATOF events, usually loaded from an ATOF JSONL exporter.
    root_parent_id:
        NAT span id to use when a Relay event has no ``parent_uuid``. Use the
        current NAT active span id to nest Relay telemetry under the adapter
        call that launched Relay.
    function_ancestry:
        NAT invocation node to attach to emitted steps. Defaults to a synthetic
        Relay bridge node.
    """

    ancestry = function_ancestry or InvocationNode(function_id="nemo-relay", function_name="nemo-relay")
    events = list(events)
    event_uuids = {event["uuid"] for event in events if isinstance(event.get("uuid"), str) and event.get("uuid")}
    start_timestamps: dict[str, float] = {}
    steps: list[IntermediateStep] = []

    for event in events:
        kind = event.get("kind")
        if kind == "scope":
            step = _scope_event_to_step(event, root_parent_id, ancestry, start_timestamps, event_uuids)
            if step is not None:
                steps.append(step)
        elif kind == "mark":
            steps.extend(_mark_event_to_steps(event, root_parent_id, ancestry, event_uuids))

    return steps


def inject_atof_jsonl(path: str | Path, *, context: Context | None = None) -> list[IntermediateStep]:
    """Load Relay ATOF JSONL and inject it into the current NAT stream."""

    context = context or Context.get()
    steps = relay_events_to_intermediate_steps(load_atof_jsonl(path),
                                               root_parent_id=context.active_span_id,
                                               function_ancestry=context.active_function)
    context.intermediate_step_manager.push_intermediate_steps(steps)
    return steps


def _scope_event_to_step(event: RelayEvent,
                         root_parent_id: str,
                         function_ancestry: InvocationNode,
                         start_timestamps: dict[str, float],
                         event_uuids: set[str]) -> IntermediateStep | None:
    event_uuid = _event_uuid(event)
    category = str(event.get("category") or "custom")
    scope_category = str(event.get("scope_category") or "")
    timestamp = _timestamp_seconds(event.get("timestamp"))

    if scope_category == "start":
        event_type = _event_types_for_category(category)[0]
        start_timestamps[event_uuid] = timestamp
        data = StreamEventData(input=event.get("data"))
        payload = IntermediateStepPayload(UUID=event_uuid,
                                          event_type=event_type,
                                          name=_event_name(event),
                                          event_timestamp=timestamp,
                                          metadata=_trace_metadata(event),
                                          data=data)
        return IntermediateStep(parent_id=_parent_id(event, root_parent_id, event_uuids),
                                function_ancestry=function_ancestry,
                                payload=payload)

    if scope_category == "end":
        event_type = _event_types_for_category(category)[1]
        start_timestamp = start_timestamps.get(event_uuid, timestamp)
        data = StreamEventData(output=event.get("data"))
        payload = IntermediateStepPayload(UUID=event_uuid,
                                          event_type=event_type,
                                          name=_event_name(event),
                                          event_timestamp=timestamp,
                                          span_event_timestamp=start_timestamp,
                                          metadata=_trace_metadata(event),
                                          data=data,
                                          usage_info=_usage_info(event))
        return IntermediateStep(parent_id=_parent_id(event, root_parent_id, event_uuids),
                                function_ancestry=function_ancestry,
                                payload=payload)

    return None


def _mark_event_to_steps(event: RelayEvent,
                         root_parent_id: str,
                         function_ancestry: InvocationNode,
                         event_uuids: set[str]) -> list[IntermediateStep]:
    event_uuid = _event_uuid(event)
    timestamp = _timestamp_seconds(event.get("timestamp"))
    name = f"mark:{_event_name(event)}"
    metadata = _trace_metadata(event)
    parent_id = _parent_id(event, root_parent_id, event_uuids)
    start = IntermediateStep(parent_id=parent_id,
                             function_ancestry=function_ancestry,
                             payload=IntermediateStepPayload(UUID=event_uuid,
                                                             event_type=IntermediateStepType.CUSTOM_START,
                                                             name=name,
                                                             event_timestamp=timestamp,
                                                             metadata=metadata,
                                                             data=StreamEventData(input=event.get("data"))))
    end = IntermediateStep(parent_id=parent_id,
                           function_ancestry=function_ancestry,
                           payload=IntermediateStepPayload(UUID=event_uuid,
                                                           event_type=IntermediateStepType.CUSTOM_END,
                                                           name=name,
                                                           event_timestamp=timestamp,
                                                           span_event_timestamp=timestamp,
                                                           metadata=metadata,
                                                           data=StreamEventData(output=event.get("data"))))
    return [start, end]


def _event_types_for_category(category: str) -> tuple[IntermediateStepType, IntermediateStepType]:
    return _CATEGORY_TO_EVENT_TYPES.get(category, (IntermediateStepType.CUSTOM_START, IntermediateStepType.CUSTOM_END))


def _event_uuid(event: RelayEvent) -> str:
    uuid = event.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise ValueError("Relay event is missing a non-empty string uuid")
    return uuid


def _parent_id(event: RelayEvent, root_parent_id: str, event_uuids: set[str]) -> str:
    parent_uuid = event.get("parent_uuid")
    if isinstance(parent_uuid, str) and parent_uuid in event_uuids:
        return parent_uuid
    return root_parent_id


def _event_name(event: RelayEvent) -> str:
    name = event.get("name")
    return name if isinstance(name, str) and name else _event_uuid(event)


def _timestamp_seconds(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(normalized).timestamp()
    return datetime.now(UTC).timestamp()


def _trace_metadata(event: RelayEvent) -> TraceMetadata:
    relay_metadata = {
        "kind": event.get("kind"),
        "category": event.get("category"),
        "scope_category": event.get("scope_category"),
        "uuid": event.get("uuid"),
        "parent_uuid": event.get("parent_uuid"),
        "attributes": event.get("attributes"),
        "category_profile": event.get("category_profile"),
        "metadata": event.get("metadata"),
        "atof_version": event.get("atof_version"),
    }
    return TraceMetadata(
        provided_metadata={
            "display_name": _event_name(event),
            "nemo_relay": {
                key: value
                for key, value in relay_metadata.items() if value is not None
            },
        })


def _usage_info(event: RelayEvent) -> UsageInfo | None:
    token_usage = _find_token_usage(event)
    if not token_usage:
        return None

    prompt_tokens = _first_int(token_usage, "prompt_tokens", "input_tokens", "prompt", "input")
    completion_tokens = _first_int(token_usage, "completion_tokens", "output_tokens", "completion", "output")
    total_tokens = _first_int(token_usage, "total_tokens", "total")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return UsageInfo(token_usage=TokenUsageBaseModel(prompt_tokens=prompt_tokens,
                                                     completion_tokens=completion_tokens,
                                                     total_tokens=total_tokens),
                     num_llm_calls=1 if event.get("category") == "llm" else 0)


def _find_token_usage(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        for key in ("token_usage", "usage", "usage_info"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
        for nested in value.values():
            found = _find_token_usage(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_token_usage(nested)
            if found:
                return found
    return None


def _first_int(values: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = values.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return 0
