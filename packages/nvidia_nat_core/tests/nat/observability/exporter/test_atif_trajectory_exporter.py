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

"""Unit tests for ATIFTrajectorySpanExporter.

Tests the batch trajectory-to-span conversion:
    ATIF Trajectory (dict) -> list[Span]
"""

import json

import pytest

from nat.data_models.span import Span
from nat.data_models.span import SpanAttributes
from nat.data_models.span import SpanKind
from nat.observability.exporter.atif_trajectory_exporter import ATIFTrajectorySpanExporter
from nat.observability.exporter.atif_trajectory_exporter import _is_terminal_agent_step


# ---------------------------------------------------------------------------
# Fixture trajectories (matching theoretical example patterns)
# ---------------------------------------------------------------------------

SIMPLE_TRAJECTORY = {
    "schema_version": "ATIF-v1.6",
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "agent": {"name": "simple-calculator-agent", "version": "1.0.0"},
    "steps": [
        {
            "step_id": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "source": "user",
            "message": "What is 3 + 4?",
        },
        {
            "step_id": 2,
            "timestamp": "2026-01-01T00:00:01Z",
            "source": "agent",
            "message": "I will calculate 3 + 4 using the calculator tool.",
            "tool_calls": [
                {
                    "tool_call_id": "call_calc_001",
                    "function_name": "calculator__add",
                    "arguments": {"a": 3, "b": 4},
                }
            ],
            "observation": {
                "results": [
                    {"source_call_id": "call_calc_001", "content": "7"}
                ]
            },
            "extra": {
                "ancestry": {
                    "function_id": "fn_simple_agent_step",
                    "function_name": "simple_calculator_agent",
                    "parent_id": None,
                    "parent_name": None,
                },
                "invocation": {
                    "start_timestamp": 1735689601.0,
                    "end_timestamp": 1735689602.5,
                    "invocation_id": "inv_simple_step_2",
                },
                "tool_ancestry": [
                    {
                        "function_id": "fn_calc_add",
                        "function_name": "calculator__add",
                        "parent_id": "fn_simple_agent_step",
                        "parent_name": "simple_calculator_agent",
                    }
                ],
                "tool_invocations": [
                    {
                        "start_timestamp": 1735689601.1,
                        "end_timestamp": 1735689601.4,
                        "invocation_id": "call_calc_001",
                    }
                ],
            },
        },
        {
            "step_id": 3,
            "timestamp": "2026-01-01T00:00:02Z",
            "source": "agent",
            "message": "The result of 3 + 4 is 7.",
            "extra": {
                "ancestry": {
                    "function_id": "fn_simple_agent_final",
                    "function_name": "simple_calculator_agent",
                    "parent_id": None,
                    "parent_name": None,
                },
                "invocation": {
                    "start_timestamp": 1735689602.5,
                    "end_timestamp": 1735689603.0,
                    "invocation_id": "inv_simple_step_3",
                },
            },
        },
    ],
}

MULTI_AGENT_TRAJECTORY = {
    "schema_version": "ATIF-v1.6",
    "session_id": "ecafea93-d983-5e70-85e4-bf54b251d1b0",
    "agent": {"name": "ChatAssistant", "version": "1.2.0"},
    "steps": [
        {
            "step_id": 1,
            "timestamp": "2026-02-23T12:06:18Z",
            "source": "user",
            "message": "{}",
        },
        {
            "step_id": 2,
            "timestamp": "2026-02-23T12:06:18Z",
            "source": "agent",
            "message": "Hello! I'm your assistant.",
            "extra": {
                "ancestry": {
                    "function_id": "fn_chat_step_2",
                    "function_name": "ChatAssistant",
                    "parent_id": None,
                    "parent_name": None,
                },
                "invocation": {
                    "start_timestamp": 1740312378.0,
                    "end_timestamp": 1740312386.0,
                },
                "tool_ancestry": [],
                "tool_invocations": None,
            },
        },
        {
            "step_id": 3,
            "timestamp": "2026-02-23T12:06:26Z",
            "source": "user",
            "message": "How does Carpenter work?",
        },
        {
            "step_id": 4,
            "timestamp": "2026-02-23T12:06:26Z",
            "source": "agent",
            "message": "I'll look that up.",
            "tool_calls": [
                {
                    "tool_call_id": "tooluse_p3ULs5EH",
                    "function_name": "ConfluenceAgent",
                    "arguments": {"question": "How does Carpenter work?"},
                }
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "tooluse_p3ULs5EH",
                        "content": "Carpenter is an ADK.",
                        "subagent_trajectory_ref": [
                            {
                                "session_id": "c78ad2b0-5437-512e-bb3d-2807207a4e2b",
                                "trajectory_path": None,
                            }
                        ],
                    }
                ]
            },
            "extra": {
                "ancestry": {
                    "function_id": "fn_chat_step_4",
                    "function_name": "ChatAssistant",
                    "parent_id": None,
                    "parent_name": None,
                },
                "invocation": {
                    "start_timestamp": 1740312386.0,
                    "end_timestamp": 1740312401.0,
                },
                "tool_ancestry": [
                    {
                        "function_id": "fn_confluence_delegation",
                        "function_name": "ConfluenceAgent",
                        "parent_id": "fn_chat_step_4",
                        "parent_name": "ChatAssistant",
                    }
                ],
                "tool_invocations": [
                    {
                        "start_timestamp": 1740312386.0,
                        "end_timestamp": 1740312401.0,
                    }
                ],
            },
        },
        {
            "step_id": 5,
            "timestamp": "2026-02-23T12:06:41Z",
            "source": "agent",
            "message": "Carpenter is an ADK for multi-agent applications.",
            "extra": {
                "ancestry": {
                    "function_id": "fn_chat_step_5",
                    "function_name": "ChatAssistant",
                    "parent_id": None,
                    "parent_name": None,
                },
                "invocation": {
                    "start_timestamp": 1740312401.0,
                    "end_timestamp": 1740312401.0,
                },
                "tool_ancestry": [],
                "tool_invocations": None,
            },
        },
    ],
    "subagent_trajectories": [
        {
            "schema_version": "ATIF-v1.6",
            "session_id": "c78ad2b0-5437-512e-bb3d-2807207a4e2b",
            "agent": {"name": "ConfluenceAgent", "version": "1.2.0"},
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-02-23T12:06:30Z",
                    "source": "user",
                    "message": "How does Carpenter work?",
                },
                {
                    "step_id": 2,
                    "timestamp": "2026-02-23T12:06:30Z",
                    "source": "agent",
                    "message": "Read the page.",
                    "tool_calls": [
                        {
                            "tool_call_id": "tooluse_285Hqm",
                            "function_name": "ConfluenceReader",
                            "arguments": {"url": "https://wiki.example.com/Carpenter"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "tooluse_285Hqm",
                                "content": "Carpenter is an ADK.",
                            }
                        ]
                    },
                    "extra": {
                        "ancestry": {
                            "function_id": "fn_conf_step_2",
                            "function_name": "ConfluenceAgent",
                            "parent_id": None,
                            "parent_name": None,
                        },
                        "invocation": {
                            "start_timestamp": 1740312390.0,
                            "end_timestamp": 1740312393.0,
                        },
                        "tool_ancestry": [
                            {
                                "function_id": "fn_conf_reader",
                                "function_name": "ConfluenceReader",
                                "parent_id": "fn_conf_step_2",
                                "parent_name": "ConfluenceAgent",
                            }
                        ],
                        "tool_invocations": [
                            {
                                "start_timestamp": 1740312390.0,
                                "end_timestamp": 1740312390.5,
                            }
                        ],
                    },
                },
                {
                    "step_id": 3,
                    "timestamp": "2026-02-23T12:06:33Z",
                    "source": "agent",
                    "message": "Carpenter is an ADK for multi-agent apps.",
                    "extra": {
                        "ancestry": {
                            "function_id": "fn_conf_step_3",
                            "function_name": "ConfluenceAgent",
                            "parent_id": None,
                            "parent_name": None,
                        },
                        "invocation": {
                            "start_timestamp": 1740312393.0,
                            "end_timestamp": 1740312393.0,
                        },
                    },
                },
            ],
        }
    ],
}

NO_EXTRA_TRAJECTORY = {
    "schema_version": "ATIF-v1.6",
    "session_id": "00000000-0000-0000-0000-000000000001",
    "agent": {"name": "bare-agent", "version": "1.0.0"},
    "steps": [
        {"step_id": 1, "source": "user", "message": "Hello"},
        {"step_id": 2, "source": "agent", "message": "Hi there!"},
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTraceId:
    """Tests for trace ID generation."""

    def test_each_convert_produces_unique_trace_id(self):
        exporter = ATIFTrajectorySpanExporter()
        spans_a = exporter.convert(SIMPLE_TRAJECTORY)
        spans_b = exporter.convert(SIMPLE_TRAJECTORY)
        assert spans_a[0].context.trace_id != spans_b[0].context.trace_id


class TestIsTerminalAgentStep:
    """Tests for terminal agent step detection on raw dicts."""

    def test_message_only_is_terminal(self):
        assert _is_terminal_agent_step({"source": "agent", "message": "done"})

    def test_with_tools_is_not_terminal(self):
        assert not _is_terminal_agent_step(
            {"source": "agent", "message": "using tool", "tool_calls": [{}]}
        )

    def test_user_step_is_not_terminal(self):
        assert not _is_terminal_agent_step({"source": "user", "message": "hello"})

    def test_empty_message_is_not_terminal(self):
        assert not _is_terminal_agent_step({"source": "agent", "message": ""})


class TestSimpleTrajectory:
    """Tests using the EXMP-01 simple calculator pattern."""

    @pytest.fixture
    def exporter(self):
        return ATIFTrajectorySpanExporter()

    def test_produces_spans(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        assert len(spans) > 0

    def test_root_is_workflow_span(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        root = spans[0]
        assert root.attributes.get("nat.span.kind") == SpanKind.WORKFLOW.value
        assert root.name == "simple-calculator-agent"

    def test_workflow_span_has_io(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        root = spans[0]
        assert root.attributes.get(SpanAttributes.INPUT_VALUE.value) == "What is 3 + 4?"
        assert "7" in root.attributes.get(SpanAttributes.OUTPUT_VALUE.value, "")

    def test_all_spans_share_trace_id(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        root_tid = spans[0].context.trace_id
        assert root_tid > 0
        for span in spans:
            assert span.context.trace_id == root_tid

    def test_llm_span_created(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        llm_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.LLM.value
        ]
        assert len(llm_spans) >= 1

    def test_tool_span_created(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        tool_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.TOOL.value
        ]
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "calculator__add"

    def test_tool_span_has_io(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        tool_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.TOOL.value
        ]
        tool = tool_spans[0]
        assert '"a": 3' in tool.attributes.get(SpanAttributes.INPUT_VALUE.value, "")
        assert tool.attributes.get(SpanAttributes.OUTPUT_VALUE.value) == "7"

    def test_tool_span_is_child_of_llm(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        tool_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.TOOL.value
        ]
        tool = tool_spans[0]
        assert tool.parent is not None
        assert tool.parent.name == "simple_calculator_agent"

    def test_llm_spans_reparented_under_workflow(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        llm_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.LLM.value
        ]
        for llm in llm_spans:
            assert llm.parent is not None
            assert llm.parent.attributes.get("nat.span.kind") == SpanKind.WORKFLOW.value

    def test_span_timing(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        tool_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.TOOL.value
        ]
        tool = tool_spans[0]
        expected_start_ns = int(1735689601.1 * 1e9)
        expected_end_ns = int(1735689601.4 * 1e9)
        assert tool.start_time == expected_start_ns
        assert tool.end_time == expected_end_ns

    def test_session_id_as_conversation_id(self, exporter):
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        for span in spans:
            assert span.attributes.get("session.id") == SIMPLE_TRAJECTORY["session_id"]


class TestMultiAgentTrajectory:
    """Tests using the EXMP-04 multi-agent delegation pattern."""

    @pytest.fixture
    def exporter(self):
        return ATIFTrajectorySpanExporter()

    def test_produces_parent_and_subagent_spans(self, exporter):
        spans = exporter.convert(MULTI_AGENT_TRAJECTORY)
        # Should have: root workflow + 3 LLM spans + 1 tool span (parent)
        #            + subagent root workflow + 1 LLM + 1 tool + 1 terminal LLM
        assert len(spans) >= 5

    def test_subagent_spans_share_parent_trace_id(self, exporter):
        spans = exporter.convert(MULTI_AGENT_TRAJECTORY)
        root_tid = spans[0].context.trace_id
        assert root_tid > 0
        for span in spans:
            assert span.context.trace_id == root_tid

    def test_delegation_ref_tracked(self, exporter):
        spans = exporter.convert(MULTI_AGENT_TRAJECTORY)
        # The subagent's root workflow span should be linked to the tool span
        sub_workflow_spans = [
            s for s in spans
            if s.name == "ConfluenceAgent"
            and s.attributes.get("nat.span.kind") == SpanKind.WORKFLOW.value
        ]
        assert len(sub_workflow_spans) == 1
        sub_root = sub_workflow_spans[0]
        # Should have a parent (the delegating tool span)
        assert sub_root.parent is not None
        assert sub_root.parent.name == "ConfluenceAgent"

    def test_multi_turn_workflow_io(self, exporter):
        spans = exporter.convert(MULTI_AGENT_TRAJECTORY)
        root = spans[0]
        assert root.attributes.get("nat.span.kind") == SpanKind.WORKFLOW.value
        # First user message
        assert root.attributes.get(SpanAttributes.INPUT_VALUE.value) == "{}"
        # Last terminal agent message
        assert "multi-agent" in root.attributes.get(SpanAttributes.OUTPUT_VALUE.value, "")


class TestNoExtraTrajectory:
    """Tests for trajectories where agent steps lack extra metadata."""

    @pytest.fixture
    def exporter(self):
        return ATIFTrajectorySpanExporter()

    def test_produces_workflow_span(self, exporter):
        spans = exporter.convert(NO_EXTRA_TRAJECTORY)
        assert len(spans) >= 1
        root = spans[0]
        assert root.attributes.get("nat.span.kind") == SpanKind.WORKFLOW.value

    def test_workflow_has_user_input(self, exporter):
        spans = exporter.convert(NO_EXTRA_TRAJECTORY)
        root = spans[0]
        assert root.attributes.get(SpanAttributes.INPUT_VALUE.value) == "Hello"

    def test_workflow_has_agent_output(self, exporter):
        spans = exporter.convert(NO_EXTRA_TRAJECTORY)
        root = spans[0]
        assert root.attributes.get(SpanAttributes.OUTPUT_VALUE.value) == "Hi there!"


class TestTokenMetrics:
    """Tests for token count propagation."""

    @pytest.fixture
    def trajectory_with_metrics(self):
        return {
            "schema_version": "ATIF-v1.6",
            "session_id": "00000000-0000-0000-0000-000000000002",
            "agent": {"name": "metric-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Count tokens"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Tokens counted.",
                    "metrics": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    },
                    "extra": {
                        "ancestry": {
                            "function_id": "fn_metric_step",
                            "function_name": "metric_agent",
                            "parent_id": None,
                            "parent_name": None,
                        },
                        "invocation": {
                            "start_timestamp": 1000.0,
                            "end_timestamp": 1001.0,
                        },
                    },
                },
            ],
        }

    def test_token_counts_on_llm_span(self, trajectory_with_metrics):
        exporter = ATIFTrajectorySpanExporter()
        spans = exporter.convert(trajectory_with_metrics)
        llm_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.LLM.value
        ]
        assert len(llm_spans) == 1
        llm = llm_spans[0]
        assert llm.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT.value) == 100
        assert llm.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value) == 50
        assert llm.attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL.value) == 150


class TestSpanPrefix:
    """Tests for custom span prefix."""

    def test_custom_prefix(self):
        exporter = ATIFTrajectorySpanExporter(span_prefix="custom")
        spans = exporter.convert(SIMPLE_TRAJECTORY)
        root = spans[0]
        assert root.attributes.get("custom.span.kind") == SpanKind.WORKFLOW.value
        assert "custom.event_type" in root.attributes


class TestNestedToolChain:
    """Tests for nested tool ancestry (parent tool → child tool)."""

    @pytest.fixture
    def nested_trajectory(self):
        return {
            "schema_version": "ATIF-v1.6",
            "session_id": "00000000-0000-0000-0000-000000000003",
            "agent": {"name": "nested-agent", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Look up weather and convert"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "I will look up and convert.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_weather",
                            "function_name": "weather__lookup",
                            "arguments": {"city": "SF"},
                        },
                        {
                            "tool_call_id": "call_convert",
                            "function_name": "temperature__to_celsius",
                            "arguments": {"fahrenheit": 68.0},
                        },
                    ],
                    "observation": {
                        "results": [
                            {"source_call_id": "call_weather", "content": "68.0 F"},
                            {"source_call_id": "call_convert", "content": "20.0 C"},
                        ]
                    },
                    "extra": {
                        "ancestry": {
                            "function_id": "fn_agent_step",
                            "function_name": "nested_agent",
                            "parent_id": None,
                            "parent_name": None,
                        },
                        "invocation": {
                            "start_timestamp": 1000.0,
                            "end_timestamp": 1002.0,
                        },
                        "tool_ancestry": [
                            {
                                "function_id": "fn_weather",
                                "function_name": "weather__lookup",
                                "parent_id": "fn_agent_step",
                                "parent_name": "nested_agent",
                            },
                            {
                                "function_id": "fn_convert",
                                "function_name": "temperature__to_celsius",
                                "parent_id": "fn_weather",
                                "parent_name": "weather__lookup",
                            },
                        ],
                        "tool_invocations": [
                            {"start_timestamp": 1000.1, "end_timestamp": 1000.8},
                            {"start_timestamp": 1000.9, "end_timestamp": 1001.5},
                        ],
                    },
                },
                {
                    "step_id": 3,
                    "source": "agent",
                    "message": "The weather is 20.0 C.",
                    "extra": {
                        "ancestry": {
                            "function_id": "fn_agent_final",
                            "function_name": "nested_agent",
                            "parent_id": None,
                            "parent_name": None,
                        },
                        "invocation": {
                            "start_timestamp": 1002.0,
                            "end_timestamp": 1002.5,
                        },
                    },
                },
            ],
        }

    def test_nested_tools_both_created(self, nested_trajectory):
        exporter = ATIFTrajectorySpanExporter()
        spans = exporter.convert(nested_trajectory)
        tool_spans = [
            s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.TOOL.value
        ]
        assert len(tool_spans) == 2
        names = {s.name for s in tool_spans}
        assert names == {"weather__lookup", "temperature__to_celsius"}

    def test_child_tool_parents_parent_tool(self, nested_trajectory):
        exporter = ATIFTrajectorySpanExporter()
        spans = exporter.convert(nested_trajectory)
        tool_spans = {
            s.name: s for s in spans
            if s.attributes.get("nat.span.kind") == SpanKind.TOOL.value
        }
        convert_span = tool_spans["temperature__to_celsius"]
        assert convert_span.parent is not None
        assert convert_span.parent.name == "weather__lookup"
