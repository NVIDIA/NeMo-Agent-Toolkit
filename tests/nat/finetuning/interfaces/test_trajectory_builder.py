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

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import EpisodeItem
from nat.data_models.finetuning import EpisodeItemRole
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.data_models.finetuning import TrajectoryCollection
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder


class ConcreteTrajectoryBuilder(TrajectoryBuilder):
    """Concrete implementation of TrajectoryBuilder for testing."""

    def __init__(self, trajectory_builder_config: TrajectoryBuilderConfig):
        super().__init__(trajectory_builder_config)
        self.started_runs = []
        self.finalized_runs = []
        self.computed_rewards = []
        self.logged_progress = []
        self.trajectories_data = []

    async def start_run(self, run_id: str, meta: dict | None = None) -> None:
        """Initialize resources for the trajectory builder."""
        self.started_runs.append((run_id, meta))

    async def finalize(self, run_id: str, meta: dict | None = None) -> TrajectoryCollection:
        """Finalize and return constructed trajectories."""
        self.finalized_runs.append((run_id, meta))

        # Create sample trajectories
        trajectories = [[
            Trajectory(episode=[
                EpisodeItem(role=EpisodeItemRole.USER, content="test input", logprobs=None),
                EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="test output", logprobs={"test": 0.5})
            ],
                       reward=0.8,
                       shaped_rewards=[0.4, 0.4],
                       metadata={"example_id": str(i)})
        ] for i in range(len(self.trajectories_data))]

        return TrajectoryCollection(trajectories=trajectories, run_id=run_id)

    def log_progress(self, run_id: str, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log trajectory building progress."""
        self.logged_progress.append({"run_id": run_id, "metrics": metrics, "output_dir": output_dir})


class TestTrajectoryBuilder:
    """Tests for the TrajectoryBuilder interface."""

    @pytest.fixture
    def builder_config(self):
        """Create a test trajectory builder config."""
        return TrajectoryBuilderConfig(type="test_trajectory_builder", reward=RewardFunctionConfig(name="test_reward"))

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create a test finetune config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{"input": "test"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        return FinetuneConfig(run_configuration=run_config, curriculum_learning=CurriculumLearningConfig())

    @pytest.fixture
    def builder(self, builder_config):
        """Create a concrete trajectory builder instance."""
        return ConcreteTrajectoryBuilder(trajectory_builder_config=builder_config)

    async def test_builder_initialization(self, builder, builder_config):
        """Test that builder initializes with correct configuration."""
        assert builder.trajectory_builder_config == builder_config
        assert builder.run_config is None

    async def test_builder_initialize(self, builder, finetune_config):
        """Test builder initialization."""
        await builder.initialize(finetune_config)
        assert builder.run_config == finetune_config

    async def test_builder_start_run(self, builder):
        """Test starting a run."""
        meta = {"experiment": "test_experiment"}
        await builder.start_run(run_id="run_001", meta=meta)

        assert len(builder.started_runs) == 1
        assert builder.started_runs[0] == ("run_001", meta)

    async def test_builder_start_run_without_meta(self, builder):
        """Test starting a run without metadata."""
        await builder.start_run(run_id="run_001", meta=None)

        assert len(builder.started_runs) == 1
        assert builder.started_runs[0] == ("run_001", None)

    async def test_builder_finalize(self, builder):
        """Test finalizing trajectory building."""
        # Add some trajectory data
        builder.trajectories_data = [{"id": 1}, {"id": 2}]

        meta = {"total_examples": 2}
        trajectory_collection = await builder.finalize(run_id="run_001", meta=meta)

        assert isinstance(trajectory_collection, TrajectoryCollection)
        assert trajectory_collection.run_id == "run_001"
        assert len(trajectory_collection.trajectories) == 2
        assert len(builder.finalized_runs) == 1

    async def test_builder_finalize_with_empty_data(self, builder):
        """Test finalizing with no trajectory data."""
        trajectory_collection = await builder.finalize(run_id="run_001", meta=None)

        assert isinstance(trajectory_collection, TrajectoryCollection)
        assert len(trajectory_collection.trajectories) == 0

    async def test_builder_compute_reward(self, builder):
        """Test computing reward from output item."""
        output_item = MagicMock(spec=EvalOutputItem)
        output_item.score = 0.75

        reward = await builder.compute_reward(output_item, meta=None)

        assert reward == 0.75

    async def test_builder_compute_reward_with_none_score(self, builder):
        """Test computing reward when score is None."""
        output_item = MagicMock(spec=EvalOutputItem)
        output_item.score = None

        reward = await builder.compute_reward(output_item, meta=None)

        assert reward == 0.0

    async def test_builder_compute_reward_with_metadata(self, builder):
        """Test computing reward with metadata."""
        output_item = MagicMock(spec=EvalOutputItem)
        output_item.score = 0.9
        meta = {"multiplier": 2.0}

        reward = await builder.compute_reward(output_item, meta=meta)

        # Default implementation ignores metadata
        assert reward == 0.9

    def test_builder_log_progress(self, builder):
        """Test logging progress."""
        metrics = {"trajectories_built": 10, "avg_reward": 0.8}
        builder.log_progress(run_id="run_001", metrics=metrics, output_dir="/tmp/logs")

        assert len(builder.logged_progress) == 1
        assert builder.logged_progress[0]["run_id"] == "run_001"
        assert builder.logged_progress[0]["metrics"] == metrics
        assert builder.logged_progress[0]["output_dir"] == "/tmp/logs"

    @patch('nat.eval.evaluate.EvaluationRun')
    async def test_builder_run_eval(self, mock_eval_run, builder, finetune_config):
        """Test running evaluation."""
        await builder.initialize(finetune_config)

        # Mock the evaluation run
        mock_eval_output = MagicMock(spec=EvaluationRunOutput)
        mock_eval_instance = AsyncMock()
        mock_eval_instance.run_and_evaluate = AsyncMock(return_value=mock_eval_output)
        mock_eval_run.return_value = mock_eval_instance

        eval_output = await builder.run_eval()

        assert eval_output == mock_eval_output
        mock_eval_run.assert_called_once()
        mock_eval_instance.run_and_evaluate.assert_called_once()

    async def test_builder_trajectory_structure(self, builder):
        """Test that finalized trajectories have correct structure."""
        builder.trajectories_data = [{"id": 1}]

        trajectory_collection = await builder.finalize(run_id="run_001", meta=None)

        assert len(trajectory_collection.trajectories) == 1
        trajectory_group = trajectory_collection.trajectories[0]
        assert len(trajectory_group) == 1

        trajectory = trajectory_group[0]
        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.episode) == 2
        assert trajectory.episode[0].role == EpisodeItemRole.USER
        assert trajectory.episode[1].role == EpisodeItemRole.ASSISTANT
        assert trajectory.reward == 0.8
        assert trajectory.shaped_rewards == [0.4, 0.4]


class TestTrajectoryBuilderEdgeCases:
    """Tests for TrajectoryBuilder edge cases and error handling."""

    @pytest.fixture
    def builder_config(self):
        """Create a test trajectory builder config."""
        return TrajectoryBuilderConfig(type="test_trajectory_builder")

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create a test finetune config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{"input": "test"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        return FinetuneConfig(run_configuration=run_config)

    class FailingTrajectoryBuilder(TrajectoryBuilder):
        """Builder that fails during operations."""

        async def start_run(self, run_id: str, meta: dict | None = None) -> None:
            raise RuntimeError("Start run failed")

        async def finalize(self, run_id: str, meta: dict | None = None) -> TrajectoryCollection:
            raise RuntimeError("Finalize failed")

        def log_progress(self, run_id: str, metrics: dict[str, Any], output_dir: str | None = None) -> None:
            raise RuntimeError("Logging failed")

    async def test_builder_start_run_failure(self, builder_config):
        """Test handling of start_run failures."""
        builder = self.FailingTrajectoryBuilder(builder_config)

        with pytest.raises(RuntimeError, match="Start run failed"):
            await builder.start_run("run_001")

    async def test_builder_finalize_failure(self, builder_config):
        """Test handling of finalize failures."""
        builder = self.FailingTrajectoryBuilder(builder_config)

        with pytest.raises(RuntimeError, match="Finalize failed"):
            await builder.finalize("run_001")

    async def test_builder_log_progress_failure(self, builder_config):
        """Test handling of log_progress failures."""
        builder = self.FailingTrajectoryBuilder(builder_config)

        with pytest.raises(RuntimeError, match="Logging failed"):
            builder.log_progress("run_001", {})

    async def test_builder_multiple_runs(self, builder_config):
        """Test handling multiple runs sequentially."""
        builder = ConcreteTrajectoryBuilder(builder_config)

        # Start and finalize first run
        await builder.start_run("run_001")
        builder.trajectories_data = [{"id": 1}]
        collection1 = await builder.finalize("run_001")

        # Start and finalize second run
        await builder.start_run("run_002")
        builder.trajectories_data = [{"id": 2}, {"id": 3}]
        collection2 = await builder.finalize("run_002")

        assert len(builder.started_runs) == 2
        assert len(builder.finalized_runs) == 2
        assert collection1.run_id == "run_001"
        assert collection2.run_id == "run_002"
        assert len(collection1.trajectories) == 1
        assert len(collection2.trajectories) == 2

    async def test_builder_trajectory_with_logprobs(self, builder_config):
        """Test that trajectories properly handle logprobs."""
        builder = ConcreteTrajectoryBuilder(builder_config)
        builder.trajectories_data = [{"id": 1}]

        collection = await builder.finalize("run_001")
        trajectory = collection.trajectories[0][0]

        # User message should have no logprobs
        assert trajectory.episode[0].logprobs is None
        # Assistant message should have logprobs
        assert trajectory.episode[1].logprobs is not None
        assert isinstance(trajectory.episode[1].logprobs, dict)

    async def test_trajectory_builder_config_reward_field(self):
        """Test that TrajectoryBuilderConfig has reward field that can be set."""

        class TestTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="test_builder_with_reward"):
            pass

        config = TestTrajectoryBuilderConfig(reward=RewardFunctionConfig(name="test_reward"))
        assert config.reward is not None
        assert isinstance(config.reward, RewardFunctionConfig)
        assert config.reward.name == "test_reward"

    async def test_trajectory_builder_config_reward_field_default(self):
        """Test that TrajectoryBuilderConfig reward field defaults to None."""

        class TestTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="test_builder_no_reward"):
            pass

        config = TestTrajectoryBuilderConfig()
        assert config.reward is None
