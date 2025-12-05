# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pytest configuration and shared fixtures for NeMo Customizer tests."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem

# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def dpo_config():
    """Create a default DPO trajectory builder configuration."""
    from nat.plugins.customizer.dpo.config import DPOTrajectoryBuilderConfig

    return DPOTrajectoryBuilderConfig(
        custom_step_name="dpo_candidate_move",
        exhaustive_pairs=True,
        min_score_diff=0.0,
        max_pairs_per_turn=None,
        turn_id_key="turn_id",
        score_key="score",
        prompt_key="prompt",
        response_key="raw_llm_response",
        candidate_index_key="candidate_index",
        reward_from_score_diff=True,
        require_multiple_candidates=True,
    )


@pytest.fixture
def dpo_builder(dpo_config):
    """Create a DPO trajectory builder instance."""
    from nat.plugins.customizer.dpo.dpo_trajectory_builder import DPOTrajectoryBuilder

    return DPOTrajectoryBuilder(trajectory_builder_config=dpo_config)


def create_intermediate_step(
    step_name: str,
    metadata: dict[str, Any],
    step_type: IntermediateStepType = IntermediateStepType.CUSTOM_END,
) -> IntermediateStep:
    """Helper function to create an intermediate step with given metadata."""
    # Create the payload with event_type (category is derived from event_type)
    payload = IntermediateStepPayload(
        event_type=step_type,
        UUID=f"test-uuid-{metadata.get('candidate_index', 0)}",
        name=step_name,
        metadata=metadata,
    )
    # IntermediateStep requires parent_id, function_ancestry, and payload
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(
            function_id="test-function-id",
            function_name="test_function",
        ),
        payload=payload,
    )


def create_candidate_metadata(
    turn_id: str,
    candidate_index: int,
    score: float,
    prompt: str = "Test prompt",
    response: str = "Test response",
    is_selected: bool = False,
) -> dict[str, Any]:
    """Helper function to create candidate step metadata."""
    return {
        "turn_id": turn_id,
        "candidate_index": candidate_index,
        "score": score,
        "prompt": prompt,
        "raw_llm_response": response,
        "is_selected": is_selected,
    }


@pytest.fixture
def sample_candidates():
    """Create sample candidate metadata for testing."""
    return [
        create_candidate_metadata("turn_0", 0, 0.9, "Board state 1", "Move A", True),
        create_candidate_metadata("turn_0", 1, 0.7, "Board state 1", "Move B", False),
        create_candidate_metadata("turn_0", 2, 0.5, "Board state 1", "Move C", False),
    ]


@pytest.fixture
def sample_intermediate_steps(sample_candidates):
    """Create sample intermediate steps for testing."""
    return [create_intermediate_step("dpo_candidate_move", metadata) for metadata in sample_candidates]


@pytest.fixture
def mock_eval_result(sample_intermediate_steps):
    """Create a mock evaluation result with sample intermediate steps."""
    # Create EvalInputItem with trajectory
    input_item = EvalInputItem(
        id="example_1",
        input_obj={"board": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
        expected_output_obj=None,
        full_dataset_entry={},
        trajectory=sample_intermediate_steps,
    )

    # Create mock eval_input
    eval_input = EvalInput(eval_input_items=[input_item], )

    # Create mock evaluation output
    mock_output = MagicMock()
    mock_output.eval_input = eval_input

    return mock_output


@pytest.fixture
def multi_turn_candidates():
    """Create candidates across multiple turns for testing."""
    return [
        # Turn 0 candidates
        create_candidate_metadata("turn_0", 0, 0.9, "Turn 0 board", "Turn 0 Move A", True),
        create_candidate_metadata("turn_0", 1, 0.7, "Turn 0 board", "Turn 0 Move B", False),
        # Turn 1 candidates
        create_candidate_metadata("turn_1", 0, 0.8, "Turn 1 board", "Turn 1 Move A", True),
        create_candidate_metadata("turn_1", 1, 0.6, "Turn 1 board", "Turn 1 Move B", False),
        create_candidate_metadata("turn_1", 2, 0.4, "Turn 1 board", "Turn 1 Move C", False),
    ]


@pytest.fixture
def multi_turn_intermediate_steps(multi_turn_candidates):
    """Create intermediate steps for multiple turns."""
    return [create_intermediate_step("dpo_candidate_move", metadata) for metadata in multi_turn_candidates]


@pytest.fixture
def mock_multi_turn_eval_result(multi_turn_intermediate_steps):
    """Create mock evaluation result with multiple turns."""
    input_item = EvalInputItem(
        id="example_multi",
        input_obj={},
        expected_output_obj=None,
        full_dataset_entry={},
        trajectory=multi_turn_intermediate_steps,
    )

    eval_input = EvalInput(eval_input_items=[input_item], )

    mock_output = MagicMock()
    mock_output.eval_input = eval_input

    return mock_output


@pytest.fixture
def multi_example_candidates():
    """Create candidates from multiple examples for testing grouping."""
    return [
        # Example 1, Turn 0
        create_candidate_metadata("turn_0", 0, 0.9, "Ex1 T0", "Ex1 T0 Move A"),
        create_candidate_metadata("turn_0", 1, 0.7, "Ex1 T0", "Ex1 T0 Move B"),
        # Example 2, Turn 0
        create_candidate_metadata("turn_0", 0, 0.85, "Ex2 T0", "Ex2 T0 Move A"),
        create_candidate_metadata("turn_0", 1, 0.65, "Ex2 T0", "Ex2 T0 Move B"),
    ]
