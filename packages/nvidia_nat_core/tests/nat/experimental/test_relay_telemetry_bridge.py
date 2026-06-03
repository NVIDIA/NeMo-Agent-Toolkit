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

import json

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode
from nat.experimental.relay_telemetry_bridge import inject_atof_jsonl
from nat.experimental.relay_telemetry_bridge import load_atof_jsonl
from nat.experimental.relay_telemetry_bridge import relay_events_to_intermediate_steps


def _scope_event(uuid: str,
                 *,
                 category: str,
                 scope_category: str,
                 parent_uuid: str | None = None,
                 data=None,
                 metadata=None,
                 category_profile=None,
                 timestamp: str = "2026-05-29T20:00:00Z"):
    return {
        "kind": "scope",
        "atof_version": "0.1",
        "uuid": uuid,
        "parent_uuid": parent_uuid,
        "timestamp": timestamp,
        "name": f"{category}-{scope_category}",
        "scope_category": scope_category,
        "category": category,
        "data": data,
        "metadata": metadata,
        "category_profile": category_profile,
    }


def test_relay_scope_events_map_to_nat_start_end_with_parentage():
    ancestry = InvocationNode(function_id="fn-1", function_name="adapter")
    events = [
        _scope_event("agent-1",
                     category="agent",
                     scope_category="start",
                     parent_uuid="relay-gateway-root",
                     data={"prompt": "hi"}),
        _scope_event("llm-1",
                     category="llm",
                     scope_category="start",
                     parent_uuid="agent-1",
                     data={"messages": []},
                     category_profile={"model_name": "relay-model"}),
        _scope_event("llm-1",
                     category="llm",
                     scope_category="end",
                     parent_uuid="agent-1",
                     data={
                         "message": "hello", "usage": {
                             "prompt_tokens": 2, "completion_tokens": 3
                         }
                     }),
        _scope_event("agent-1", category="agent", scope_category="end", data={"done": True}),
    ]

    steps = relay_events_to_intermediate_steps(events, root_parent_id="nat-parent", function_ancestry=ancestry)

    assert [step.event_type for step in steps] == [
        IntermediateStepType.TASK_START,
        IntermediateStepType.LLM_START,
        IntermediateStepType.LLM_END,
        IntermediateStepType.TASK_END,
    ]
    assert steps[0].parent_id == "nat-parent"
    assert steps[1].parent_id == "agent-1"
    assert steps[2].payload.span_event_timestamp == steps[1].payload.event_timestamp
    assert steps[2].usage_info is not None
    assert steps[2].usage_info.token_usage.total_tokens == 5
    assert steps[1].metadata.provided_metadata["nemo_relay"]["category_profile"] == {"model_name": "relay-model"}


def test_relay_mark_events_become_zero_duration_custom_spans():
    ancestry = InvocationNode(function_id="fn-1", function_name="adapter")
    events = [
        _scope_event("agent-1", category="agent", scope_category="start"),
        {
            "kind": "mark",
            "atof_version": "0.1",
            "uuid": "mark-1",
            "parent_uuid": "agent-1",
            "timestamp": "2026-05-29T20:00:01Z",
            "name": "checkpoint",
            "category": "custom",
            "data": {
                "step": 1
            },
        },
    ]

    steps = relay_events_to_intermediate_steps(events, root_parent_id="nat-parent", function_ancestry=ancestry)

    assert [step.event_type for step in steps] == [
        IntermediateStepType.TASK_START,
        IntermediateStepType.CUSTOM_START,
        IntermediateStepType.CUSTOM_END,
    ]
    assert steps[1].name == "mark:checkpoint"
    assert steps[1].parent_id == "agent-1"
    assert steps[2].payload.span_event_timestamp == steps[1].payload.event_timestamp


def test_load_and_inject_relay_atof_jsonl(tmp_path):
    path = tmp_path / "relay-events.jsonl"
    events = [
        _scope_event("tool-1", category="tool", scope_category="start", data={"query": "weather"}),
        _scope_event("tool-1", category="tool", scope_category="end", data={"answer": "sunny"}),
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    assert load_atof_jsonl(path) == events

    ctx_state = ContextState()
    ctx_state.active_span_id_stack.set(["nat-parent"])
    ctx_state.active_function.set(InvocationNode(function_id="fn-1", function_name="adapter"))
    context = Context(ctx_state)
    captured: list[IntermediateStep] = []
    context.intermediate_step_manager.subscribe(captured.append)

    injected = inject_atof_jsonl(path, context=context)

    assert injected == captured
    assert [step.event_type for step in captured] == [IntermediateStepType.TOOL_START, IntermediateStepType.TOOL_END]
    assert captured[0].parent_id == "nat-parent"


def test_load_relay_atof_jsonl_skips_malformed_lines(tmp_path):
    path = tmp_path / "relay-events.jsonl"
    event = _scope_event("tool-1", category="tool", scope_category="start")
    path.write_text(json.dumps(event) + "\nnot json\n[]\n", encoding="utf-8")

    assert load_atof_jsonl(path) == [event]
