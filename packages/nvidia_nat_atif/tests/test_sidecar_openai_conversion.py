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
"""Regression tests for OpenAI-compatible sidecar ATOF streams.

The NeMo-Flow gateway sidecar captures LLM request/response traffic but does
not yet emit first-class tool scope events. Tool results are still recoverable
from the next OpenAI request history as ``role=tool`` messages.
"""

from __future__ import annotations

from typing import Any

from nat.atof import ScopeEvent
from nat.atof.scripts.atof_to_atif_converter import convert


def _llm_start(uuid: str, timestamp: str, messages: list[dict[str, Any]]) -> ScopeEvent:
    return ScopeEvent(
        scope_category="start",
        uuid=uuid,
        parent_uuid="gateway-001",
        timestamp=timestamp,
        name="openai.chat_completions",
        category="llm",
        category_profile={"model_name": "qwen3.6:35b"},
        data={"content": {
            "messages": messages, "model": "qwen3.6:35b"
        }},
    )


def _llm_end(
    uuid: str,
    timestamp: str,
    *,
    content: str,
    usage: dict[str, int],
    tool_calls: list[dict[str, Any]] | None = None,
) -> ScopeEvent:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return ScopeEvent(
        scope_category="end",
        uuid=uuid,
        parent_uuid="gateway-001",
        timestamp=timestamp,
        name="openai.chat_completions",
        category="llm",
        category_profile={"model_name": "qwen3.6:35b"},
        data={
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
            }],
            "model": "qwen3.6:35b",
            "usage": usage,
        },
    )


def _lossy_hermes_api_hook_pair() -> list[ScopeEvent]:
    return [
        ScopeEvent(
            scope_category="start",
            uuid="lossy-llm-001",
            parent_uuid="gateway-001",
            timestamp="2026-05-12T22:00:00.500Z",
            name="custom",
            category="llm",
            category_profile={"model_name": "qwen3.6:35b"},
            data={
                "content": {
                    "api_call_count": 1,
                    "message_count": 2,
                    "model": "qwen3.6:35b",
                    "fidelity": {
                        "provider_payload_exact": False, "source": "hermes_pre_api_request"
                    },
                },
                "headers": {},
            },
            metadata={
                "fidelity_source": "hermes_api_hooks",
                "provider_payload_exact": False,
            },
        ),
        ScopeEvent(
            scope_category="end",
            uuid="lossy-llm-001",
            parent_uuid="gateway-001",
            timestamp="2026-05-12T22:00:00.900Z",
            name="custom",
            category="llm",
            category_profile={"model_name": "qwen3.6:35b"},
            data={
                "api_call_count": 1,
                "assistant_content_chars": 12,
                "message_count": 2,
                "model": "qwen3.6:35b",
                "usage": {
                    "prompt_tokens": 99, "completion_tokens": 3, "total_tokens": 102
                },
            },
            metadata={
                "fidelity_source": "hermes_api_hooks",
                "provider_payload_exact": False,
            },
        ),
    ]


def _sidecar_stream_without_tool_scopes() -> list[ScopeEvent]:
    system = {"role": "system", "content": "You are Hermes Agent."}
    user = {"role": "user", "content": "Find the answer."}
    assistant_tool = {
        "role":
            "assistant",
        "content":
            "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_files", "arguments": '{"pattern":"needle"}'
            },
        }],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "search_files",
        "content": '{"total_count":1,"files":["needle.py"]}',
    }
    orphan_user = {"role": "user", "content": "Review the conversation and update memory."}

    return [
        ScopeEvent(
            scope_category="start",
            uuid="gateway-001",
            parent_uuid="external-parent-not-in-stream",
            timestamp="2026-05-12T22:00:00Z",
            name="gateway",
            category="agent",
            metadata={
                "agent": "hermes-nemoflow", "session_id": "gateway-gateway", "source": "harbor"
            },
        ),
        _llm_start("llm-001", "2026-05-12T22:00:01Z", [system, user]),
        _llm_end(
            "llm-001",
            "2026-05-12T22:00:02Z",
            content="",
            tool_calls=assistant_tool["tool_calls"],
            usage={
                "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12
            },
        ),
        _llm_start("llm-002", "2026-05-12T22:00:03Z", [system, user, assistant_tool, tool_result]),
        _llm_end(
            "llm-002",
            "2026-05-12T22:00:04Z",
            content="Done.",
            usage={
                "prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23
            },
        ),
        _llm_start("llm-orphan", "2026-05-12T22:00:05Z", [system, user, assistant_tool, tool_result, orphan_user]),
        ScopeEvent(
            scope_category="end",
            uuid="gateway-001",
            parent_uuid="external-parent-not-in-stream",
            timestamp="2026-05-12T22:00:06Z",
            name="gateway",
            category="agent",
            metadata={
                "agent": "hermes-nemoflow", "session_id": "gateway-gateway", "source": "harbor"
            },
        ),
    ]


def test_sidecar_openai_history_recovers_observations_metrics_and_metadata() -> None:
    trajectory = convert(_sidecar_stream_without_tool_scopes())

    assert trajectory.session_id == "gateway-gateway"
    assert trajectory.agent.name == "hermes-nemoflow"
    assert trajectory.agent.model_name == "qwen3.6:35b"

    assert [step.source for step in trajectory.steps] == ["system", "user", "agent", "agent"]
    assert [step.message for step in trajectory.steps] == [
        "You are Hermes Agent.",
        "Find the answer.",
        "[tool call]",
        "Done.",
    ]

    tool_step = trajectory.steps[2]
    assert tool_step.tool_calls is not None
    assert tool_step.tool_calls[0].tool_call_id == "call_1"
    assert tool_step.tool_calls[0].function_name == "search_files"
    assert tool_step.tool_calls[0].arguments == {"pattern": "needle"}
    assert tool_step.observation is not None
    assert tool_step.observation.results[0].source_call_id == "call_1"
    assert tool_step.observation.results[0].content == '{"total_count":1,"files":["needle.py"]}'

    assert tool_step.metrics is not None
    assert tool_step.metrics.prompt_tokens == 10
    assert tool_step.metrics.completion_tokens == 2
    assert tool_step.metrics.extra == {"total_tokens": 12}

    final_step = trajectory.steps[3]
    assert final_step.metrics is not None
    assert final_step.metrics.prompt_tokens == 20
    assert final_step.metrics.completion_tokens == 3

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_steps == 4
    assert trajectory.final_metrics.total_prompt_tokens == 30
    assert trajectory.final_metrics.total_completion_tokens == 5
    assert trajectory.final_metrics.extra == {"total_tokens": 35}


def test_sidecar_openai_conversion_ignores_lossy_hermes_hook_summaries() -> None:
    events = _sidecar_stream_without_tool_scopes()
    events[1:1] = _lossy_hermes_api_hook_pair()

    trajectory = convert(events)

    assert [step.source for step in trajectory.steps] == ["system", "user", "agent", "agent"]
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 30
    assert trajectory.final_metrics.total_completion_tokens == 5


def test_sidecar_openai_conversion_ignores_agent_end_tool_cleanup() -> None:
    events = _sidecar_stream_without_tool_scopes()
    events.extend([
        ScopeEvent(
            scope_category="start",
            uuid="fallback-tool-001",
            parent_uuid="gateway-001",
            timestamp="2026-05-12T22:00:05.200Z",
            name="patch",
            category="tool",
            category_profile={"tool_call_id": "tool-fallback-001"},
            data={
                "path": "/testbed/file.py",
                "old_string": "before",
                "new_string": "after",
            },
            metadata={
                "hook_event_name": "pre_tool_call",
                "tool_correlation_status": "agent_fallback",
            },
        ),
        ScopeEvent(
            scope_category="end",
            uuid="fallback-tool-001",
            parent_uuid="gateway-001",
            timestamp="2026-05-12T22:00:05.800Z",
            name="patch",
            category="tool",
            category_profile={"tool_call_id": "tool-fallback-001"},
            data={"status": "closed_by_agent_end"},
            metadata={
                "hook_event_name": "pre_tool_call",
                "status": "closed_by_agent_end",
                "tool_correlation_status": "agent_fallback",
            },
        ),
    ])

    trajectory = convert(events)

    source_call_ids = [
        result.source_call_id
        for step in trajectory.steps
        if step.observation is not None
        for result in step.observation.results
    ]
    assert "call_1" in source_call_ids
    assert "tool-fallback-001" not in source_call_ids


def test_sidecar_openai_conversion_keeps_usage_only_empty_assistant_turn() -> None:
    events = _sidecar_stream_without_tool_scopes()
    events.insert(
        -1,
        _llm_start("llm-empty", "2026-05-12T22:00:05.500Z", [{
            "role": "user", "content": "No-op."
        }]),
    )
    events.insert(
        -1,
        _llm_end(
            "llm-empty",
            "2026-05-12T22:00:05.900Z",
            content="",
            usage={
                "prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8
            },
        ),
    )

    trajectory = convert(events)

    assert trajectory.steps[-1].source == "agent"
    assert trajectory.steps[-1].message == ""
    assert trajectory.steps[-1].metrics is not None
    assert trajectory.steps[-1].metrics.prompt_tokens == 7
    assert trajectory.steps[-1].metrics.completion_tokens == 1


def test_sidecar_openai_conversion_skips_empty_assistant_noop_turn() -> None:
    events = _sidecar_stream_without_tool_scopes()
    events.insert(
        -1,
        _llm_start("llm-noop", "2026-05-12T22:00:05.500Z", [{
            "role": "assistant", "content": "Previous assistant message."
        }]),
    )
    events.insert(
        -1,
        ScopeEvent(
            scope_category="end",
            uuid="llm-noop",
            parent_uuid="gateway-001",
            timestamp="2026-05-12T22:00:05.900Z",
            name="openai.chat_completions",
            category="llm",
            category_profile={"model_name": "qwen3.6:35b"},
            data={"role": "assistant", "content": None},
        ),
    )

    trajectory = convert(events)

    assert [step.source for step in trajectory.steps] == ["system", "user", "agent", "agent"]
    assert [step.message for step in trajectory.steps] == [
        "You are Hermes Agent.",
        "Find the answer.",
        "[tool call]",
        "Done.",
    ]
