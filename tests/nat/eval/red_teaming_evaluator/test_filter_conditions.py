# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition


@pytest.fixture(name="create_intermediate_step")
def fixture_create_intermediate_step():
    """
    Factory fixture to create an IntermediateStep with customizable parameters.

    Args:
        event_type: The type of event (IntermediateStepType)
        name: The name of the step (e.g., tool or function name)
        parent_id: Parent ID (defaults to "root")
        function_name: Function name for invocation node (defaults to name or "test_function")
        function_id: Function ID for invocation node (defaults to "test-function-id")

    Returns:
        A function that creates IntermediateStep objects
    """

    def _create_step(
        event_type: IntermediateStepType,
        name: str | None = None,
        parent_id: str = "root",
        function_name: str | None = None,
        function_id: str = "test-function-id",
        input_data: str | None = None,
        output_data: str | None = None,
    ) -> IntermediateStep:
        """Create an IntermediateStep with the specified parameters."""
        payload = IntermediateStepPayload(
            event_type=event_type,
            name=name,
            data=StreamEventData(input=input_data, output=output_data) if input_data or output_data else None,
        )
        return IntermediateStep(
            parent_id=parent_id,
            function_ancestry=InvocationNode(
                function_name=function_name or name or "test_function",
                function_id=function_id,
            ),
            payload=payload,
        )

    return _create_step


@pytest.fixture(name="sample_trajectory")
def fixture_sample_trajectory(create_intermediate_step):
    """
    Fixture that creates a sample trajectory with various event types and names.

    Contains:
    - LLM_START (name: "llm_model")
    - LLM_END (name: "llm_model")
    - TOOL_START (name: "calculator")
    - TOOL_END (name: "calculator")
    - TOOL_START (name: "search_tool")
    - TOOL_END (name: "search_tool")
    - FUNCTION_START (name: "process_data")
    - FUNCTION_END (name: "process_data")
    """
    return [
        create_intermediate_step(IntermediateStepType.LLM_START, name="llm_model"),
        create_intermediate_step(IntermediateStepType.LLM_END, name="llm_model"),
        create_intermediate_step(IntermediateStepType.TOOL_START, name="calculator"),
        create_intermediate_step(IntermediateStepType.TOOL_END, name="calculator"),
        create_intermediate_step(IntermediateStepType.TOOL_START, name="search_tool"),
        create_intermediate_step(IntermediateStepType.TOOL_END, name="search_tool"),
        create_intermediate_step(IntermediateStepType.FUNCTION_START, name="process_data"),
        create_intermediate_step(IntermediateStepType.FUNCTION_END, name="process_data"),
    ]


@pytest.fixture(name="trajectory_with_none_names")
def fixture_trajectory_with_none_names(create_intermediate_step):
    """
    Fixture that creates a trajectory with some steps having None names.
    """
    return [
        create_intermediate_step(IntermediateStepType.LLM_START, name=None),
        create_intermediate_step(IntermediateStepType.LLM_END, name="llm_model"),
        create_intermediate_step(IntermediateStepType.TOOL_START, name=None),
        create_intermediate_step(IntermediateStepType.TOOL_END, name="calculator"),
    ]


class TestIntermediateStepsFilterCondition:
    """Test suite for IntermediateStepsFilterCondition."""

    def test_filter_by_event_type_enum(self, sample_trajectory):
        """Test filtering by event_type using enum value."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type=IntermediateStepType.TOOL_END,
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == 2, "Should return 2 TOOL_END steps"
        assert all(step.event_type == IntermediateStepType.TOOL_END for step in filtered)
        assert all(step.payload.name in ["calculator", "search_tool"] for step in filtered)

    def test_filter_by_event_type_string(self, sample_trajectory):
        """Test filtering by event_type using string value."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type="TOOL_END",
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == 2, "Should return 2 TOOL_END steps"
        assert all(step.event_type == IntermediateStepType.TOOL_END for step in filtered)

    def test_filter_by_payload_name(self, sample_trajectory):
        """Test filtering by payload.name."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            payload_name="calculator",
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == 2, "Should return 2 calculator steps (START and END)"
        assert all(step.payload.name == "calculator" for step in filtered)
        assert all(step.event_type in [IntermediateStepType.TOOL_START, IntermediateStepType.TOOL_END]
                   for step in filtered)

    def test_filter_by_event_type_and_payload_name(self, sample_trajectory):
        """Test filtering by both event_type and payload_name."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type=IntermediateStepType.TOOL_END,
            payload_name="calculator",
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == 1, "Should return 1 calculator TOOL_END step"
        assert filtered[0].event_type == IntermediateStepType.TOOL_END
        assert filtered[0].payload.name == "calculator"

    def test_filter_no_conditions(self, sample_trajectory):
        """Test that filtering with no conditions returns all steps."""
        filter_condition = IntermediateStepsFilterCondition(name="test_filter")

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == len(sample_trajectory), "Should return all steps"
        assert filtered == sample_trajectory, "Should return steps in original order"

    def test_filter_empty_trajectory(self):
        """Test filtering an empty trajectory."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type=IntermediateStepType.TOOL_END,
        )

        filtered = filter_condition.filter_trajectory([])

        assert len(filtered) == 0, "Should return empty list"

    def test_filter_no_matches(self, sample_trajectory):
        """Test filtering when no steps match the conditions."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type=IntermediateStepType.TOOL_END,
            payload_name="nonexistent_tool",
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == 0, "Should return empty list when no matches"

    def test_filter_payload_name_with_none_values(self, trajectory_with_none_names):
        """Test filtering by payload_name when some steps have None names."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            payload_name="calculator",
        )

        filtered = filter_condition.filter_trajectory(trajectory_with_none_names)

        assert len(filtered) == 1, "Should return 1 calculator step"
        assert filtered[0].payload.name == "calculator"
        assert filtered[0].event_type == IntermediateStepType.TOOL_END

    def test_filter_multiple_tools_same_event_type(self, sample_trajectory):
        """Test filtering multiple tools with the same event type."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type=IntermediateStepType.TOOL_START,
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        assert len(filtered) == 2, "Should return 2 TOOL_START steps"
        assert all(step.event_type == IntermediateStepType.TOOL_START for step in filtered)
        assert set(step.payload.name for step in filtered) == {"calculator", "search_tool"}

    def test_filter_preserves_order(self, sample_trajectory):
        """Test that filtering preserves the original order of steps."""
        filter_condition = IntermediateStepsFilterCondition(
            name="test_filter",
            event_type=IntermediateStepType.TOOL_END,
        )

        filtered = filter_condition.filter_trajectory(sample_trajectory)

        # Check that order is preserved (calculator comes before search_tool in original)
        assert filtered[0].payload.name == "calculator"
        assert filtered[1].payload.name == "search_tool"

    def test_filter_condition_name_field(self):
        """Test that the name field is properly set."""
        filter_condition = IntermediateStepsFilterCondition(
            name="my_custom_filter",
            event_type=IntermediateStepType.LLM_END,
        )

        assert filter_condition.name == "my_custom_filter"
        assert filter_condition.event_type == IntermediateStepType.LLM_END
