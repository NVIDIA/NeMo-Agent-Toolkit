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

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import TrajectoryCollection
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.invocation_node import InvocationNode
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.plugins.openpipe.config import ARTTrajectoryBuilderConfig
from nat.plugins.openpipe.trajectory_builder import ARTTrajectoryBuilder


class TestARTTrajectoryBuilder:
    """Comprehensive tests for ARTTrajectoryBuilder implementation."""

    @pytest.fixture
    def builder_config(self):
        """Create test trajectory builder configuration."""
        return ARTTrajectoryBuilderConfig(num_generations=2, reward=RewardFunctionConfig(name="test_reward"))

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create test finetune configuration."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{"input": "test"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        finetune_config = FinetuneConfig(run_configuration=run_config,
                                         curriculum_learning=CurriculumLearningConfig(),
                                         reward_function=RewardFunctionConfig(name="test_reward"))

        # Add target_functions directly to the config for testing
        finetune_config.target_functions = ["test_function"]

        return finetune_config

    @pytest.fixture
    def builder(self, builder_config):
        """Create ARTTrajectoryBuilder instance."""
        return ARTTrajectoryBuilder(trajectory_builder_config=builder_config)

    def test_builder_initialization(self, builder, builder_config):
        """Test that builder initializes with correct configuration."""
        assert builder.trajectory_builder_config == builder_config
        assert builder.evaluation_runs == {}
        assert builder.num_generations == 2

    async def test_builder_initialize(self, builder, finetune_config):
        """Test builder initialization with finetune config."""
        await builder.initialize(finetune_config)

        assert builder.run_config == finetune_config
        assert builder.trajectory_builder_config.reward == finetune_config.reward_function

    async def test_start_run(self, builder):
        """Test starting evaluation runs."""
        # Mock run_eval
        mock_eval_output = MagicMock()
        builder.run_eval = AsyncMock(return_value=mock_eval_output)

        await builder.start_run(run_id="test_run", meta={"epoch": 0})

        assert "test_run" in builder.evaluation_runs
        assert len(builder.evaluation_runs["test_run"]) == 2  # num_generations

        # Verify tasks were created
        tasks = builder.evaluation_runs["test_run"]
        for task in tasks:
            assert isinstance(task, asyncio.Task)

    async def test_start_run_duplicate(self, builder):
        """Test starting duplicate run raises error."""
        builder.evaluation_runs["test_run"] = []

        with pytest.raises(ValueError, match="Run test_run is already in progress"):
            await builder.start_run(run_id="test_run")

    async def test_finalize_with_trajectories(self, builder, finetune_config):
        """Test finalizing and building trajectories from evaluation results."""
        await builder.initialize(finetune_config)

        # Create mock intermediate steps
        step1 = MagicMock(spec=IntermediateStep)
        step1.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        step2 = MagicMock(spec=IntermediateStep)
        step2.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        # Create mock evaluation results
        eval_item = EvalOutputItem(id="item_1", score=0.8, reasoning="Good answer")

        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test input",
                                   expected_output_obj="Test output",
                                   full_dataset_entry={},
                                   trajectory=[step1, step2])

        # Create mock evaluation output
        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        # Mock parse_to_openai_messages
        mock_openai_messages = [{
            "role": "user", "content": "Test question"
        }, {
            "role": "assistant", "content": "Test answer", "logprobs": {
                "test": 0.5
            }
        }]

        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   return_value=mock_openai_messages):
            # Create tasks that return the mock evaluation output
            task1 = AsyncMock(return_value=mock_eval_output)
            task2 = AsyncMock(return_value=mock_eval_output)

            # Mock asyncio.gather to return our results
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=[mock_eval_output, mock_eval_output]):
                builder.evaluation_runs["test_run"] = [task1, task2]

                # Finalize and get trajectories
                collection = await builder.finalize(run_id="test_run", meta={"epoch": 0})

        assert isinstance(collection, TrajectoryCollection)
        assert collection.run_id == "test_run"
        assert len(collection.trajectories) > 0

        # Verify cleanup
        assert "test_run" not in builder.evaluation_runs

    async def test_finalize_no_reward_results(self, builder, finetune_config):
        """Test finalizing when no reward results found."""
        await builder.initialize(finetune_config)

        # Create mock evaluation output without reward results
        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("other_metric", MagicMock(eval_output_items=[]))]

        # Create tasks
        task = AsyncMock(return_value=mock_eval_output)

        with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                   new_callable=AsyncMock,
                   return_value=[mock_eval_output]):
            builder.evaluation_runs["test_run"] = [task]

            collection = await builder.finalize(run_id="test_run")

        assert len(collection.trajectories) == 0

    async def test_finalize_unknown_run(self, builder):
        """Test finalizing unknown run raises error."""
        with pytest.raises(ValueError, match="No evaluation runs found"):
            await builder.finalize(run_id="unknown_run")

    async def test_finalize_single_generation(self, builder, finetune_config):
        """Test finalizing with single generation configuration."""
        builder.trajectory_builder_config.num_generations = 1
        await builder.initialize(finetune_config)

        # Create mock trajectory
        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        eval_item = EvalOutputItem(id="item_1", score=0.9, reasoning="Excellent")

        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test",
                                   expected_output_obj="Output",
                                   full_dataset_entry={},
                                   trajectory=[step])

        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        mock_openai_messages = [{
            "role": "user", "content": "Test"
        }, {
            "role": "assistant", "content": "Response", "logprobs": {
                "test": 0.5
            }
        }]

        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   return_value=mock_openai_messages):
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=[mock_eval_output]):
                builder.evaluation_runs["test_run"] = [AsyncMock()]

                collection = await builder.finalize(run_id="test_run")

        # With single generation, trajectories should be flat
        assert len(collection.trajectories) == 1
        assert isinstance(collection.trajectories[0], list)

    async def test_finalize_no_target_function_trajectory(self, builder, finetune_config):
        """Test finalizing when no trajectory matches target function."""
        await builder.initialize(finetune_config)

        # Create step with different function
        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="other_id", function_name="other_function")

        eval_item = EvalOutputItem(id="item_1", score=0.8, reasoning="test")
        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test",
                                   expected_output_obj=None,
                                   full_dataset_entry={},
                                   trajectory=[step])

        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                   new_callable=AsyncMock,
                   return_value=[mock_eval_output]):
            builder.evaluation_runs["test_run"] = [AsyncMock()]

            collection = await builder.finalize(run_id="test_run")

        # Should have no trajectories as none matched target function
        assert len(collection.trajectories) == 0

    async def test_finalize_invalid_episode(self, builder, finetune_config):
        """Test finalizing with invalid episode (no assistant with logprobs)."""
        await builder.initialize(finetune_config)

        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        eval_item = EvalOutputItem(id="item_1", score=0.8, reasoning="test")
        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test",
                                   expected_output_obj=None,
                                   full_dataset_entry={},
                                   trajectory=[step])

        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        # Return messages without assistant logprobs
        mock_openai_messages = [
            {
                "role": "user", "content": "Test"
            },
            {
                "role": "assistant", "content": "Response"
            }  # No logprobs
        ]

        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   return_value=mock_openai_messages):
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=[mock_eval_output]):
                builder.evaluation_runs["test_run"] = [AsyncMock()]

                collection = await builder.finalize(run_id="test_run")

        # Should have no trajectories as assistant has no logprobs
        assert len(collection.trajectories) == 0

    async def test_finalize_parse_error(self, builder, finetune_config):
        """Test handling of parse errors during trajectory construction."""
        await builder.initialize(finetune_config)

        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        eval_item = EvalOutputItem(id="item_1", score=0.8, reasoning="test")
        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test",
                                   expected_output_obj=None,
                                   full_dataset_entry={},
                                   trajectory=[step])

        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        # Make parse_to_openai_messages raise ValueError
        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   side_effect=ValueError("Parse error")):
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=[mock_eval_output]):
                builder.evaluation_runs["test_run"] = [AsyncMock()]

                collection = await builder.finalize(run_id="test_run")

        # Should have no trajectories due to parse error
        assert len(collection.trajectories) == 0

    def test_log_progress(self, builder, tmp_path):
        """Test logging trajectory building progress."""
        builder.trajectory_builder_config.num_generations = 3

        metrics = {"num_trajectories": 10, "avg_reward": 0.75}

        output_dir = tmp_path / "logs"
        builder.log_progress(run_id="test_run", metrics=metrics, output_dir=str(output_dir))

        # Check log file was created
        log_file = output_dir / "trajectory_builder_test_run.jsonl"
        assert log_file.exists()

        # Verify log content
        with open(log_file) as f:
            log_entry = json.loads(f.readline())
            assert log_entry["run_id"] == "test_run"
            assert log_entry["num_generations"] == 3
            assert log_entry["num_trajectories"] == 10
            assert log_entry["avg_reward"] == 0.75

    def test_num_generations_property(self, builder):
        """Test num_generations property."""
        assert builder.num_generations == builder.trajectory_builder_config.num_generations

    async def test_task_callback_on_success(self, builder):
        """Test task callback when evaluation succeeds."""
        mock_eval_output = MagicMock()

        # Create a custom mock task to control callback
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = None
        task.result.return_value = mock_eval_output

        callbacks = []

        def add_callback(cb):
            callbacks.append(cb)
            # Simulate immediate completion
            cb(task)

        task.add_done_callback = add_callback

        with patch('asyncio.create_task', return_value=task):
            builder.run_eval = AsyncMock(return_value=mock_eval_output)
            await builder.start_run(run_id="test_run")

        # Verify callback was added
        assert len(callbacks) == builder.num_generations

    async def test_task_callback_on_failure(self, builder):
        """Test task callback when evaluation fails."""
        # Create a custom mock task to control callback
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = Exception("Eval failed")

        callbacks = []

        def add_callback(cb):
            callbacks.append(cb)
            # Simulate failure
            cb(task)

        task.add_done_callback = add_callback

        with patch('asyncio.create_task', return_value=task):
            builder.run_eval = AsyncMock(side_effect=Exception("Eval failed"))
            await builder.start_run(run_id="test_run")

        # Verify callback was added and handled exception
        assert len(callbacks) == builder.num_generations

    async def test_task_callback_on_cancellation(self, builder):
        """Test task callback when evaluation is cancelled."""
        # Create a custom mock task to control callback
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True

        callbacks = []

        def add_callback(cb):
            callbacks.append(cb)
            # Simulate cancellation
            cb(task)

        task.add_done_callback = add_callback

        with patch('asyncio.create_task', return_value=task):
            builder.run_eval = AsyncMock()
            await builder.start_run(run_id="test_run")

        # Verify callback was added
        assert len(callbacks) == builder.num_generations

    async def test_finalize_groups_by_example_id(self, builder, finetune_config):
        """Test that trajectories are properly grouped by example ID."""
        await builder.initialize(finetune_config)

        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        # Create multiple items with same ID (should be grouped)
        eval_items = [
            EvalOutputItem(id="item_1", score=0.7, reasoning="Gen 1"),
            EvalOutputItem(id="item_1", score=0.8, reasoning="Gen 2"),
        ]

        input_items = [
            EvalInputItem(id="item_1",
                          input_obj="Test",
                          expected_output_obj=None,
                          full_dataset_entry={},
                          trajectory=[step]),
            EvalInputItem(id="item_1",
                          input_obj="Test",
                          expected_output_obj=None,
                          full_dataset_entry={},
                          trajectory=[step]),
        ]

        # Create two evaluation outputs (simulating 2 generations)
        mock_eval_outputs = []
        for i in range(2):
            mock_output = MagicMock()
            mock_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_items[i]]))]
            mock_output.eval_input.eval_input_items = [input_items[i]]
            mock_eval_outputs.append(mock_output)

        mock_openai_messages = [{
            "role": "user", "content": "Test"
        }, {
            "role": "assistant", "content": "Response", "logprobs": {
                "test": 0.5
            }
        }]

        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   return_value=mock_openai_messages):
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=mock_eval_outputs):
                builder.evaluation_runs["test_run"] = [AsyncMock(), AsyncMock()]

                collection = await builder.finalize(run_id="test_run")

        # Should have trajectories grouped by ID
        assert len(collection.trajectories) > 0
        # All trajectories in first group should have same ID
        if collection.trajectories:
            first_group = collection.trajectories[0]
            if isinstance(first_group, list) and first_group:
                assert all(hasattr(t, 'metadata') for t in first_group)

    async def test_finalize_with_tool_messages(self, builder, finetune_config):
        """Test handling of tool/function messages in episodes."""
        await builder.initialize(finetune_config)

        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        eval_item = EvalOutputItem(id="item_1", score=0.8, reasoning="test")
        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test",
                                   expected_output_obj=None,
                                   full_dataset_entry={},
                                   trajectory=[step])

        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        # Include tool message
        mock_openai_messages = [{
            "role": "user", "content": "Use tool"
        }, {
            "role": "tool", "content": "Tool result", "tool_call_id": "call_123"
        }, {
            "role": "assistant", "content": "Based on tool", "logprobs": {
                "test": 0.5
            }
        }]

        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   return_value=mock_openai_messages):
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=[mock_eval_output]):
                builder.evaluation_runs["test_run"] = [AsyncMock()]

                collection = await builder.finalize(run_id="test_run")

        # Should create trajectory with tool message
        assert len(collection.trajectories) > 0

    async def test_compute_reward(self, builder):
        """Test computing reward from output item."""
        output_item = MagicMock(spec=EvalOutputItem)
        output_item.score = 0.85

        reward = await builder.compute_reward(output_item, meta={"test": "meta"})

        assert reward == 0.85

    async def test_finalize_skips_short_episodes(self, builder, finetune_config):
        """Test that episodes with less than 2 messages are skipped."""
        await builder.initialize(finetune_config)

        step = MagicMock(spec=IntermediateStep)
        step.function_ancestry = InvocationNode(function_id="test_id", function_name="test_function")

        eval_item = EvalOutputItem(id="item_1", score=0.8, reasoning="test")
        input_item = EvalInputItem(id="item_1",
                                   input_obj="Test",
                                   expected_output_obj=None,
                                   full_dataset_entry={},
                                   trajectory=[step])

        mock_eval_output = MagicMock()
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[eval_item]))]
        mock_eval_output.eval_input.eval_input_items = [input_item]

        # Return single message (too short)
        mock_openai_messages = [{"role": "user", "content": "Test"}]

        with patch('nat.plugins.openpipe.trajectory_builder.parse_to_openai_messages',
                   return_value=mock_openai_messages):
            with patch('nat.plugins.openpipe.trajectory_builder.asyncio.gather',
                       new_callable=AsyncMock,
                       return_value=[mock_eval_output]):
                builder.evaluation_runs["test_run"] = [AsyncMock()]

                collection = await builder.finalize(run_id="test_run")

        # Should have no trajectories due to short episode
        assert len(collection.trajectories) == 0
