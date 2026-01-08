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
"""Comprehensive tests for DPO Trajectory Builder implementation."""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import DPOItem
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import TTCEventData
from nat.data_models.invocation_node import InvocationNode
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.plugins.customizer.dpo.trajectory_builder import CandidateStep
from nat.plugins.customizer.dpo.trajectory_builder import DPOTrajectoryBuilder
from nat.plugins.customizer.dpo.trajectory_builder import PreferencePair


# Helper functions (inline to avoid import path issues)
def create_candidate_ttc_data(
    turn_id: str,
    candidate_index: int,
    score: float,
    prompt: str | list[dict[str, str]] = "Test prompt",
    response: str = "Test response",
) -> TTCEventData:
    """Helper function to create TTCEventData for a candidate."""
    return TTCEventData(
        turn_id=turn_id,
        turn_index=0,
        candidate_index=candidate_index,
        score=score,
        input=prompt,
        output=response,
    )


def create_intermediate_step(
    step_name: str,
    ttc_data: TTCEventData,
    metadata: dict | None = None,
    step_type: IntermediateStepType = IntermediateStepType.TTC_END,
) -> IntermediateStep:
    """Helper function to create an intermediate step with TTCEventData."""
    payload = IntermediateStepPayload(
        event_type=step_type,
        UUID=f"test-uuid-{ttc_data.candidate_index or 0}",
        name=step_name,
        data=ttc_data,
        metadata=metadata or {},
    )
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(
            function_id="test-function-id",
            function_name="test_function",
        ),
        payload=payload,
    )


class TestCandidateStep:
    """Tests for CandidateStep dataclass."""

    def test_candidate_step_creation(self):
        """Test creating a CandidateStep instance."""
        candidate = CandidateStep(
            example_id="ex_1",
            turn_id="turn_0",
            candidate_index=0,
            prompt="Test prompt",
            response="Test response",
            score=0.85,
            raw_metadata={"key": "value"},
        )

        assert candidate.example_id == "ex_1"
        assert candidate.turn_id == "turn_0"
        assert candidate.candidate_index == 0
        assert candidate.prompt == "Test prompt"
        assert candidate.response == "Test response"
        assert candidate.score == 0.85
        assert candidate.raw_metadata == {"key": "value"}

    def test_candidate_step_default_metadata(self):
        """Test CandidateStep default raw_metadata."""
        candidate = CandidateStep(
            example_id="ex_1",
            turn_id="turn_0",
            candidate_index=0,
            prompt="Test",
            response="Response",
            score=0.5,
        )

        assert candidate.raw_metadata == {}

    def test_candidate_step_with_openai_messages(self):
        """Test CandidateStep with OpenAI message list as prompt."""
        from nat.data_models.finetuning import OpenAIMessage

        messages = [
            OpenAIMessage(role="system", content="You are helpful."),
            OpenAIMessage(role="user", content="Hello"),
        ]

        candidate = CandidateStep(
            example_id="ex_1",
            turn_id="turn_0",
            candidate_index=0,
            prompt=messages,
            response="Hi there!",
            score=0.8,
        )

        assert isinstance(candidate.prompt, list)
        assert len(candidate.prompt) == 2


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""

    def test_preference_pair_creation(self):
        """Test creating a PreferencePair instance."""
        pair = PreferencePair(
            example_id="ex_1",
            turn_id="turn_0",
            prompt="Test prompt",
            chosen_response="Better response",
            rejected_response="Worse response",
            chosen_score=0.9,
            rejected_score=0.5,
            score_diff=0.4,
            chosen_index=0,
            rejected_index=1,
            metadata={"extra": "info"},
        )

        assert pair.example_id == "ex_1"
        assert pair.turn_id == "turn_0"
        assert pair.prompt == "Test prompt"
        assert pair.chosen_response == "Better response"
        assert pair.rejected_response == "Worse response"
        assert pair.chosen_score == 0.9
        assert pair.rejected_score == 0.5
        assert pair.score_diff == 0.4
        assert pair.chosen_index == 0
        assert pair.rejected_index == 1
        assert pair.metadata == {"extra": "info"}


class TestDPOTrajectoryBuilder:
    """Comprehensive tests for DPOTrajectoryBuilder implementation."""

    def test_builder_initialization(self, dpo_builder, dpo_config):
        """Test that builder initializes with correct configuration."""
        assert dpo_builder.config == dpo_config
        assert dpo_builder.evaluation_runs == {}
        assert dpo_builder._metrics == {}

    def test_builder_config_reference(self, dpo_builder, dpo_config):
        """Test that trajectory_builder_config is set correctly."""
        assert dpo_builder.trajectory_builder_config == dpo_config

    # =========================================================================
    # start_run tests
    # =========================================================================

    async def test_start_run(self, dpo_builder):
        """Test starting an evaluation run."""
        dpo_builder.run_eval = AsyncMock(return_value=MagicMock())

        await dpo_builder.start_run(run_id="test_run", meta={"epoch": 0})

        assert "test_run" in dpo_builder.evaluation_runs
        assert isinstance(dpo_builder.evaluation_runs["test_run"], asyncio.Task)

    async def test_start_run_duplicate(self, dpo_builder):
        """Test starting duplicate run raises error."""
        dpo_builder.evaluation_runs["test_run"] = MagicMock()

        with pytest.raises(ValueError, match="Run test_run is already in progress"):
            await dpo_builder.start_run(run_id="test_run")

    async def test_start_run_callback_on_success(self, dpo_builder):
        """Test task callback when evaluation succeeds."""
        mock_eval_output = MagicMock()
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = None
        task.result.return_value = mock_eval_output

        callbacks = []
        task.add_done_callback = lambda cb: callbacks.append(cb) or cb(task)

        with patch("asyncio.create_task", return_value=task):
            dpo_builder.run_eval = AsyncMock(return_value=mock_eval_output)
            await dpo_builder.start_run(run_id="test_run")

        assert len(callbacks) == 1

    async def test_start_run_callback_on_failure(self, dpo_builder):
        """Test task callback when evaluation fails."""
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = Exception("Eval failed")

        callbacks = []
        task.add_done_callback = lambda cb: callbacks.append(cb) or cb(task)

        with patch("asyncio.create_task", return_value=task):
            dpo_builder.run_eval = AsyncMock(side_effect=Exception("Eval failed"))
            await dpo_builder.start_run(run_id="test_run")

        assert len(callbacks) == 1

    async def test_start_run_callback_on_cancellation(self, dpo_builder):
        """Test task callback when evaluation is cancelled."""
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True

        callbacks = []
        task.add_done_callback = lambda cb: callbacks.append(cb) or cb(task)

        with patch("asyncio.create_task", return_value=task):
            dpo_builder.run_eval = AsyncMock()
            await dpo_builder.start_run(run_id="test_run")

        assert len(callbacks) == 1

    # =========================================================================
    # finalize tests
    # =========================================================================

    async def test_finalize_unknown_run(self, dpo_builder):
        """Test finalizing unknown run raises error."""
        with pytest.raises(ValueError, match="No evaluation run found"):
            await dpo_builder.finalize(run_id="unknown_run")

    async def test_finalize_with_trajectories(self, dpo_builder, mock_eval_result):
        """Test finalizing and building trajectories from eval results."""

        async def return_result():
            return mock_eval_result

        task = asyncio.create_task(return_result())
        await asyncio.sleep(0)
        dpo_builder.evaluation_runs["test_run"] = task

        collection = await dpo_builder.finalize(run_id="test_run", meta={})

        assert isinstance(collection, TrajectoryCollection)
        assert collection.run_id == "test_run"
        assert len(collection.trajectories) > 0
        assert "test_run" not in dpo_builder.evaluation_runs

    async def test_finalize_empty_result(self, dpo_builder):
        """Test finalizing when no candidates found."""
        mock_output = MagicMock()
        mock_output.eval_input.eval_input_items = []

        async def return_result():
            return mock_output

        task = asyncio.create_task(return_result())
        await asyncio.sleep(0)
        dpo_builder.evaluation_runs["test_run"] = task

        collection = await dpo_builder.finalize(run_id="test_run")

        assert len(collection.trajectories) == 0

    async def test_finalize_metrics_tracking(self, dpo_builder, mock_eval_result):
        """Test that metrics are tracked during finalization."""

        async def return_result():
            return mock_eval_result

        task = asyncio.create_task(return_result())
        await asyncio.sleep(0)
        dpo_builder.evaluation_runs["test_run"] = task

        await dpo_builder.finalize(run_id="test_run")

        assert "total_turns" in dpo_builder._metrics
        assert "total_candidates" in dpo_builder._metrics
        assert "total_pairs" in dpo_builder._metrics
        assert "total_trajectories" in dpo_builder._metrics

    # =========================================================================
    # _is_target_step tests
    # =========================================================================

    def test_is_target_step_matching(self, dpo_builder):
        """Test identifying matching TTC_END steps."""
        ttc_data = create_candidate_ttc_data("t0", 0, 0.5, "prompt", "response")
        step = create_intermediate_step(
            "dpo_candidate_move",
            ttc_data,
            step_type=IntermediateStepType.TTC_END,
        )

        assert dpo_builder._is_target_step(step) is True

    def test_is_target_step_wrong_name(self, dpo_builder):
        """Test rejecting steps with wrong name."""
        ttc_data = create_candidate_ttc_data("t0", 0, 0.5, "prompt", "response")
        step = create_intermediate_step(
            "other_step",
            ttc_data,
            step_type=IntermediateStepType.TTC_END,
        )

        assert dpo_builder._is_target_step(step) is False

    def test_is_target_step_wrong_category(self, dpo_builder):
        """Test rejecting steps with wrong category."""
        ttc_data = create_candidate_ttc_data("t0", 0, 0.5, "prompt", "response")
        step = create_intermediate_step(
            "dpo_candidate_move",
            ttc_data,
            step_type=IntermediateStepType.LLM_END,
        )

        assert dpo_builder._is_target_step(step) is False

    def test_is_target_step_wrong_type(self, dpo_builder):
        """Test rejecting steps with wrong type (START instead of END)."""
        ttc_data = create_candidate_ttc_data("t0", 0, 0.5, "prompt", "response")
        step = create_intermediate_step(
            "dpo_candidate_move",
            ttc_data,
            step_type=IntermediateStepType.TTC_START,
        )

        assert dpo_builder._is_target_step(step) is False

    # =========================================================================
    # _parse_candidate tests
    # =========================================================================

    def test_parse_candidate_success(self, dpo_builder):
        """Test successfully parsing a candidate from a TTC step."""
        ttc_data = create_candidate_ttc_data("turn_0", 0, 0.85, "Test prompt", "Test response")
        step = create_intermediate_step("dpo_candidate_move", ttc_data)

        candidate = dpo_builder._parse_candidate("ex_1", step)

        assert candidate is not None
        assert candidate.example_id == "ex_1"
        assert candidate.turn_id == "turn_0"
        assert candidate.candidate_index == 0
        assert candidate.score == 0.85
        assert candidate.prompt == "Test prompt"
        assert candidate.response == "Test response"

    def test_parse_candidate_missing_turn_id(self, dpo_builder):
        """Test parsing fails when turn_id is missing."""
        ttc_data = TTCEventData(
            turn_id=None,  # Missing turn_id
            score=0.5,
            input="test",
            output="response",
        )
        step = create_intermediate_step("dpo_candidate_move", ttc_data)

        candidate = dpo_builder._parse_candidate("ex_1", step)

        assert candidate is None

    def test_parse_candidate_missing_score(self, dpo_builder):
        """Test parsing fails when score is missing."""
        ttc_data = TTCEventData(
            turn_id="t0",
            score=None,  # Missing score
            input="test",
            output="response",
        )
        step = create_intermediate_step("dpo_candidate_move", ttc_data)

        candidate = dpo_builder._parse_candidate("ex_1", step)

        assert candidate is None

    def test_parse_candidate_no_data(self, dpo_builder):
        """Test parsing fails when data is None."""
        ttc_data = create_candidate_ttc_data("t0", 0, 0.5)
        step = create_intermediate_step("dpo_candidate_move", ttc_data)
        step.payload.data = None

        candidate = dpo_builder._parse_candidate("ex_1", step)

        assert candidate is None

    def test_parse_candidate_with_openai_messages(self, dpo_builder):
        """Test parsing candidate with OpenAI message list as input."""
        messages = [
            {
                "role": "system", "content": "You are helpful."
            },
            {
                "role": "user", "content": "Hello"
            },
        ]
        ttc_data = TTCEventData(
            turn_id="turn_0",
            candidate_index=0,
            score=0.8,
            input=messages,
            output="Hi there!",
        )
        step = create_intermediate_step("dpo_candidate_move", ttc_data)

        candidate = dpo_builder._parse_candidate("ex_1", step)

        assert candidate is not None
        assert isinstance(candidate.prompt, list)
        assert len(candidate.prompt) == 2
        assert candidate.prompt[0].role == "system"

    # =========================================================================
    # _extract_prompt tests
    # =========================================================================

    def test_extract_prompt_string(self, dpo_builder):
        """Test extracting string prompt."""
        result = dpo_builder._extract_prompt("simple prompt")
        assert result == "simple prompt"

    def test_extract_prompt_none(self, dpo_builder):
        """Test extracting None returns empty string."""
        result = dpo_builder._extract_prompt(None)
        assert result == ""

    def test_extract_prompt_openai_messages(self, dpo_builder):
        """Test extracting OpenAI message list."""
        messages = [
            {
                "role": "system", "content": "System"
            },
            {
                "role": "user", "content": "User"
            },
        ]
        result = dpo_builder._extract_prompt(messages)

        assert isinstance(result, list)
        assert len(result) == 2

    # =========================================================================
    # _collect_candidates tests
    # =========================================================================

    def test_collect_candidates(self, dpo_builder, mock_eval_result):
        """Test collecting and grouping candidates by turn."""
        candidates_by_turn = dpo_builder._collect_candidates(mock_eval_result)

        assert len(candidates_by_turn) == 1
        turn_key = list(candidates_by_turn.keys())[0]
        assert "example_1" in turn_key
        assert "turn_0" in turn_key
        assert len(candidates_by_turn[turn_key]) == 3

    def test_collect_candidates_multi_turn(self, dpo_builder, mock_multi_turn_eval_result):
        """Test collecting candidates from multiple turns."""
        candidates_by_turn = dpo_builder._collect_candidates(mock_multi_turn_eval_result)

        assert len(candidates_by_turn) == 2

    def test_collect_candidates_filters_non_target_steps(self, dpo_builder):
        """Test that non-target steps are filtered out."""
        ttc_data1 = create_candidate_ttc_data("turn_0", 0, 0.9)
        ttc_data2 = create_candidate_ttc_data("turn_0", 1, 0.7)

        steps = [
            create_intermediate_step("dpo_candidate_move", ttc_data1),
            create_intermediate_step("other_step", ttc_data2),
        ]

        input_item = EvalInputItem(
            id="ex_1",
            input_obj={},
            expected_output_obj=None,
            full_dataset_entry={},
            trajectory=steps,
        )

        mock_output = MagicMock()
        mock_output.eval_input = EvalInput(eval_input_items=[input_item])

        candidates_by_turn = dpo_builder._collect_candidates(mock_output)

        total_candidates = sum(len(c) for c in candidates_by_turn.values())
        assert total_candidates == 1

    # =========================================================================
    # _generate_preference_pairs tests
    # =========================================================================

    def test_generate_exhaustive_pairs(self, dpo_builder, sample_ttc_data):
        """Test exhaustive pair generation (all pairwise comparisons)."""
        candidates = [
            CandidateStep(
                example_id="ex_1",
                turn_id="turn_0",
                candidate_index=i,
                prompt=str(data.input),
                response=str(data.output),
                score=data.score,
            ) for i, data in enumerate(sample_ttc_data)
        ]

        candidates_by_turn = {"ex_1::turn_0": candidates}
        pairs = dpo_builder._generate_preference_pairs(candidates_by_turn)

        # 3 candidates = 3 pairs: (0>1), (0>2), (1>2)
        assert len(pairs) == 3

        # Check pairs are sorted by score_diff descending
        for i in range(len(pairs) - 1):
            assert pairs[i].score_diff >= pairs[i + 1].score_diff

    def test_generate_best_vs_worst_pairs(self, dpo_config, sample_ttc_data):
        """Test best vs worst pair generation."""
        dpo_config.exhaustive_pairs = False
        builder = DPOTrajectoryBuilder(dpo_config)

        candidates = [
            CandidateStep(
                example_id="ex_1",
                turn_id="turn_0",
                candidate_index=i,
                prompt=str(data.input),
                response=str(data.output),
                score=data.score,
            ) for i, data in enumerate(sample_ttc_data)
        ]

        candidates_by_turn = {"ex_1::turn_0": candidates}
        pairs = builder._generate_preference_pairs(candidates_by_turn)

        assert len(pairs) == 1
        assert pairs[0].chosen_score == 0.9
        assert pairs[0].rejected_score == 0.5

    def test_generate_pairs_min_score_diff_filter(self, dpo_config):
        """Test that pairs below min_score_diff are filtered."""
        dpo_config.min_score_diff = 0.3
        builder = DPOTrajectoryBuilder(dpo_config)

        candidates = [
            CandidateStep("ex_1", "t0", 0, "p", "r1", 0.6),
            CandidateStep("ex_1", "t0", 1, "p", "r2", 0.5),
            CandidateStep("ex_1", "t0", 2, "p", "r3", 0.2),
        ]

        candidates_by_turn = {"ex_1::t0": candidates}
        pairs = builder._generate_preference_pairs(candidates_by_turn)

        assert len(pairs) == 2
        for pair in pairs:
            assert pair.score_diff >= 0.3

    def test_generate_pairs_max_pairs_per_turn(self, dpo_config):
        """Test max_pairs_per_turn limit."""
        dpo_config.max_pairs_per_turn = 2
        builder = DPOTrajectoryBuilder(dpo_config)

        candidates = [
            CandidateStep("ex_1", "t0", 0, "p", "r1", 0.9),
            CandidateStep("ex_1", "t0", 1, "p", "r2", 0.7),
            CandidateStep("ex_1", "t0", 2, "p", "r3", 0.5),
            CandidateStep("ex_1", "t0", 3, "p", "r4", 0.3),
        ]

        candidates_by_turn = {"ex_1::t0": candidates}
        pairs = builder._generate_preference_pairs(candidates_by_turn)

        assert len(pairs) == 2

    def test_generate_pairs_single_candidate_skip(self, dpo_builder):
        """Test that single candidate turns are skipped."""
        candidates = [CandidateStep("ex_1", "t0", 0, "p", "r1", 0.9)]

        candidates_by_turn = {"ex_1::t0": candidates}
        pairs = dpo_builder._generate_preference_pairs(candidates_by_turn)

        assert len(pairs) == 0
        assert dpo_builder._metrics.get("skipped_single_candidate", 0) > 0

    def test_generate_pairs_single_candidate_allowed(self, dpo_config):
        """Test single candidate turns when require_multiple_candidates=False."""
        dpo_config.require_multiple_candidates = False
        builder = DPOTrajectoryBuilder(dpo_config)

        candidates = [CandidateStep("ex_1", "t0", 0, "p", "r1", 0.9)]

        candidates_by_turn = {"ex_1::t0": candidates}
        pairs = builder._generate_preference_pairs(candidates_by_turn)

        assert len(pairs) == 0

    # =========================================================================
    # _build_trajectories tests
    # =========================================================================

    def test_build_trajectories(self, dpo_builder):
        """Test building trajectories from preference pairs."""
        pairs = [
            PreferencePair(
                example_id="ex_1",
                turn_id="t0",
                prompt="Test prompt",
                chosen_response="Good response",
                rejected_response="Bad response",
                chosen_score=0.9,
                rejected_score=0.5,
                score_diff=0.4,
                chosen_index=0,
                rejected_index=1,
            )
        ]

        trajectories = dpo_builder._build_trajectories(pairs)

        assert len(trajectories) == 1
        traj = trajectories[0]

        assert isinstance(traj, Trajectory)
        assert len(traj.episode) == 1

        # Check DPOItem
        dpo_item = traj.episode[0]
        assert isinstance(dpo_item, DPOItem)
        assert dpo_item.prompt == "Test prompt"
        assert dpo_item.chosen_response == "Good response"
        assert dpo_item.rejected_response == "Bad response"

        # Check reward (score_diff by default)
        assert traj.reward == 0.4

        # Check metadata
        assert traj.metadata["dpo_type"] == "preference_pair"
        assert traj.metadata["score_diff"] == 0.4

    def test_build_trajectories_reward_from_chosen_score(self, dpo_config):
        """Test reward computation from chosen score instead of diff."""
        dpo_config.reward_from_score_diff = False
        builder = DPOTrajectoryBuilder(dpo_config)

        pairs = [PreferencePair(
            "ex_1",
            "t0",
            "prompt",
            "chosen",
            "rejected",
            0.9,
            0.5,
            0.4,
            0,
            1,
        )]

        trajectories = builder._build_trajectories(pairs)

        assert trajectories[0].reward == 0.9

    def test_build_trajectories_with_openai_messages(self, dpo_builder):
        """Test building trajectories with OpenAI message prompt."""
        from nat.data_models.finetuning import OpenAIMessage

        messages = [
            OpenAIMessage(role="system", content="You are helpful."),
            OpenAIMessage(role="user", content="Hello"),
        ]

        pairs = [
            PreferencePair(
                example_id="ex_1",
                turn_id="t0",
                prompt=messages,
                chosen_response="Good response",
                rejected_response="Bad response",
                chosen_score=0.9,
                rejected_score=0.5,
                score_diff=0.4,
                chosen_index=0,
                rejected_index=1,
            )
        ]

        trajectories = dpo_builder._build_trajectories(pairs)

        dpo_item = trajectories[0].episode[0]
        assert isinstance(dpo_item.prompt, list)
        assert len(dpo_item.prompt) == 2

    # =========================================================================
    # _group_by_example tests
    # =========================================================================

    def test_group_by_example(self, dpo_builder):
        """Test grouping trajectories by example ID."""
        trajectories = [
            Trajectory(episode=[], reward=0.5, metadata={"example_id": "ex_1"}),
            Trajectory(episode=[], reward=0.6, metadata={"example_id": "ex_1"}),
            Trajectory(episode=[], reward=0.7, metadata={"example_id": "ex_2"}),
        ]

        grouped = dpo_builder._group_by_example(trajectories)

        assert len(grouped) == 2
        ex1_group = next(g for g in grouped if g[0].metadata["example_id"] == "ex_1")
        ex2_group = next(g for g in grouped if g[0].metadata["example_id"] == "ex_2")

        assert len(ex1_group) == 2
        assert len(ex2_group) == 1

    def test_group_by_example_unknown_id(self, dpo_builder):
        """Test grouping with missing example_id uses 'unknown'."""
        trajectories = [
            Trajectory(episode=[], reward=0.5, metadata={}),
        ]

        grouped = dpo_builder._group_by_example(trajectories)

        assert len(grouped) == 1

    # =========================================================================
    # log_progress tests
    # =========================================================================

    def test_log_progress(self, dpo_builder, tmp_path):
        """Test logging trajectory building progress."""
        dpo_builder._metrics = {
            "total_pairs": 10,
            "total_trajectories": 10,
        }

        metrics = {"custom_metric": 42}
        output_dir = tmp_path / "logs"

        dpo_builder.log_progress(
            run_id="test_run",
            metrics=metrics,
            output_dir=str(output_dir),
        )

        log_file = output_dir / "dpo_trajectory_builder_test_run.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            log_entry = json.loads(f.readline())
            assert log_entry["run_id"] == "test_run"
            assert log_entry["custom_metric"] == 42
            assert log_entry["total_pairs"] == 10
            assert "config" in log_entry
            assert log_entry["config"]["ttc_step_name"] == "dpo_candidate_move"

    def test_log_progress_default_output_dir(self, dpo_builder):
        """Test log_progress with default output directory."""
        dpo_builder._metrics = {}

        dpo_builder.log_progress(run_id="test_run", metrics={})

    def test_log_progress_appends_to_file(self, dpo_builder, tmp_path):
        """Test that log_progress appends to existing file."""
        dpo_builder._metrics = {"total_pairs": 5}
        output_dir = tmp_path / "logs"

        dpo_builder.log_progress(run_id="test_run", metrics={"epoch": 1}, output_dir=str(output_dir))
        dpo_builder.log_progress(run_id="test_run", metrics={"epoch": 2}, output_dir=str(output_dir))

        log_file = output_dir / "dpo_trajectory_builder_test_run.jsonl"
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 2


class TestDPOTrajectoryBuilderIntegration:
    """Integration tests for the full DPO trajectory building pipeline."""

    async def test_full_pipeline(self, dpo_builder, mock_eval_result):
        """Test the complete pipeline from start_run to finalize."""

        async def mock_run_eval():
            return mock_eval_result

        dpo_builder.run_eval = mock_run_eval

        await dpo_builder.start_run(run_id="integration_test")
        assert "integration_test" in dpo_builder.evaluation_runs

        collection = await dpo_builder.finalize(run_id="integration_test")

        assert isinstance(collection, TrajectoryCollection)
        assert collection.run_id == "integration_test"
        assert len(collection.trajectories) > 0

        # Verify trajectories have DPOItem episodes
        for group in collection.trajectories:
            for traj in group:
                assert traj.metadata.get("dpo_type") == "preference_pair"
                assert len(traj.episode) == 1
                assert isinstance(traj.episode[0], DPOItem)

    async def test_multi_turn_pipeline(self, dpo_builder, mock_multi_turn_eval_result):
        """Test pipeline with multiple turns."""

        async def mock_run_eval():
            return mock_multi_turn_eval_result

        dpo_builder.run_eval = mock_run_eval

        await dpo_builder.start_run(run_id="multi_turn_test")
        collection = await dpo_builder.finalize(run_id="multi_turn_test")

        total_trajectories = sum(len(g) for g in collection.trajectories)
        assert total_trajectories > 0
        assert dpo_builder._metrics["total_turns"] == 2

    async def test_pipeline_with_custom_config(self, dpo_config):
        """Test pipeline with custom configuration."""
        dpo_config.exhaustive_pairs = False
        dpo_config.min_score_diff = 0.1
        dpo_config.max_pairs_per_turn = 1

        builder = DPOTrajectoryBuilder(dpo_config)

        ttc_data_list = [
            create_candidate_ttc_data("turn_0", 0, 0.9, "Prompt", "Best"),
            create_candidate_ttc_data("turn_0", 1, 0.5, "Prompt", "Worst"),
            create_candidate_ttc_data("turn_0", 2, 0.7, "Prompt", "Middle"),
        ]

        steps = [create_intermediate_step("dpo_candidate_move", ttc_data) for ttc_data in ttc_data_list]

        input_item = EvalInputItem(
            id="ex_1",
            input_obj={},
            expected_output_obj=None,
            full_dataset_entry={},
            trajectory=steps,
        )

        mock_output = MagicMock()
        mock_output.eval_input = EvalInput(eval_input_items=[input_item])

        async def mock_run_eval():
            return mock_output

        builder.run_eval = mock_run_eval

        await builder.start_run(run_id="custom_config_test")
        collection = await builder.finalize(run_id="custom_config_test")

        total_trajectories = sum(len(g) for g in collection.trajectories)
        assert total_trajectories == 1

        traj = collection.trajectories[0][0]
        assert traj.metadata["chosen_score"] == 0.9
        assert traj.metadata["rejected_score"] == 0.5
