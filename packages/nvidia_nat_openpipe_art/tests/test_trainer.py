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

import json
import uuid
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.eval.config import EvaluationRunOutput
from nat.plugins.openpipe.config import ARTTrainerConfig
from nat.plugins.openpipe.trainer import ARTTrainer


class TestARTTrainer:
    """Comprehensive tests for ARTTrainer implementation."""

    @pytest.fixture
    def trainer_config(self):
        """Create test trainer configuration."""
        return ARTTrainerConfig(reward=RewardFunctionConfig(name="test_reward"))

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

        return FinetuneConfig(run_configuration=run_config,
                              curriculum_learning=CurriculumLearningConfig(enabled=False),
                              reward_function=RewardFunctionConfig(name="test_reward"),
                              output_dir=str(tmp_path / "output"))

    @pytest.fixture
    def trainer(self, trainer_config):
        """Create ARTTrainer instance."""
        return ARTTrainer(trainer_config=trainer_config)

    async def test_trainer_initialization(self, trainer, trainer_config):
        """Test that trainer initializes with correct configuration."""
        assert trainer.trainer_config == trainer_config
        assert trainer._job_refs == []
        assert trainer._run_id is None
        assert trainer._reward_history == []
        assert trainer._validation_history == []

    async def test_trainer_initialize(self, trainer, finetune_config):
        """Test trainer initialization process."""
        # Mock trajectory builder and adapter
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)

        with patch.object(uuid, 'uuid4', return_value=MagicMock(hex="abcd1234")):
            await trainer.initialize(finetune_config)

        assert trainer.run_config == finetune_config
        assert trainer._run_id.startswith("art_run_")
        assert trainer._run_id == "art_run_abcd1234"
        mock_builder.initialize.assert_called_once_with(finetune_config)
        mock_adapter.initialize.assert_called_once_with(finetune_config)

    async def test_run_epoch_with_trajectories(self, trainer, finetune_config):
        """Test running a single epoch with trajectories."""
        # Create mock trajectories
        mock_trajectory = MagicMock(spec=Trajectory)
        mock_trajectory.reward = 0.8
        trajectory_collection = TrajectoryCollection(trajectories=[[mock_trajectory, mock_trajectory]],
                                                     run_id="test_run")

        # Mock trajectory builder
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=trajectory_collection)

        # Mock trainer adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()
        mock_job_ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        mock_adapter.submit = AsyncMock(return_value=mock_job_ref)

        # Bind components before initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Run epoch
        job_ref = await trainer.run_epoch(epoch=0, run_id="test_run")

        assert job_ref == mock_job_ref
        assert job_ref in trainer._job_refs
        mock_builder.start_run.assert_called_once()
        mock_builder.finalize.assert_called_once()
        mock_adapter.submit.assert_called_once()

    async def test_run_epoch_without_trajectories(self, trainer, finetune_config):
        """Test running epoch when no trajectories are collected."""
        # Empty trajectory collection
        empty_collection = TrajectoryCollection(trajectories=[], run_id="test_run")

        # Mock trajectory builder
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=empty_collection)

        # Mock trainer adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind components before initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Run epoch
        job_ref = await trainer.run_epoch(epoch=0, run_id="test_run")

        assert job_ref is None
        mock_builder.start_run.assert_called_once()
        mock_builder.finalize.assert_called_once()

    async def test_run_multiple_epochs(self, trainer, finetune_config):
        """Test running multiple epochs."""
        # Mock trajectory builder
        mock_trajectory = MagicMock(spec=Trajectory)
        mock_trajectory.reward = 0.8

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()

        # Mock trainer adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()
        mock_adapter.submit = AsyncMock()
        mock_adapter.wait_until_complete = AsyncMock()

        # Bind components before initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Now set up the returns
        trajectory_collection = TrajectoryCollection(trajectories=[[mock_trajectory]], run_id=trainer._run_id)
        mock_builder.finalize = AsyncMock(return_value=trajectory_collection)

        mock_job_ref = TrainingJobRef(run_id=trainer._run_id, backend="openpipe-art")
        mock_status = TrainingJobStatus(run_id=trainer._run_id,
                                        backend="openpipe-art",
                                        status=TrainingStatusEnum.COMPLETED)
        mock_adapter.submit = AsyncMock(return_value=mock_job_ref)
        mock_adapter.wait_until_complete = AsyncMock(return_value=mock_status)

        # Run training
        statuses = await trainer.run(num_epochs=3)

        assert len(statuses) == 3
        assert all(s.status == TrainingStatusEnum.COMPLETED for s in statuses)
        assert mock_builder.start_run.call_count == 3
        assert mock_adapter.submit.call_count == 3

    async def test_run_with_failed_epoch(self, trainer, finetune_config):
        """Test handling of failed training epoch."""
        # Mock trajectory builder with exception
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock(side_effect=Exception("Test error"))

        # Mock trainer adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind components before initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Run training
        statuses = await trainer.run(num_epochs=2)

        assert len(statuses) == 1
        assert statuses[0].status == TrainingStatusEnum.FAILED
        assert "Test error" in statuses[0].message

    async def test_run_validation_evaluation(self, trainer, finetune_config, tmp_path):
        """Test running validation evaluation."""
        validation_dataset = tmp_path / "validation.jsonl"
        validation_dataset.write_text('{"input": "test"}')
        finetune_config.run_configuration.validation_dataset = str(validation_dataset)

        # Mock trajectory builder
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.run_eval = AsyncMock()
        mock_builder.run_config = MagicMock()
        mock_builder.run_config.run_configuration = finetune_config.run_configuration

        # Mock trainer adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind components before initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Mock evaluation output
        mock_eval_output = MagicMock(spec=EvaluationRunOutput)
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[mock_metric, mock_metric]))]
        mock_builder.run_eval = AsyncMock(return_value=mock_eval_output)

        # Run validation
        metrics = await trainer.run_validation_evaluation(epoch=0, run_id="test_run")

        assert metrics["epoch"] == 0
        assert metrics["dataset_type"] == "validation"
        assert metrics["avg_reward"] == 0.85
        assert metrics["min_reward"] == 0.85
        assert metrics["max_reward"] == 0.85
        assert metrics["num_examples"] == 2

    async def test_get_metrics(self, trainer):
        """Test getting metrics for a run."""
        trainer._run_id = "test_run"

        # Add some job refs
        job_ref1 = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        job_ref2 = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        trainer._job_refs = [job_ref1, job_ref2]

        # Mock trainer adapter
        mock_status = TrainingJobStatus(run_id="test_run", backend="openpipe-art", status=TrainingStatusEnum.COMPLETED)
        mock_adapter = MagicMock()
        mock_adapter.status = AsyncMock(return_value=mock_status)
        trainer.trainer_adapter = mock_adapter

        # Get metrics
        metrics = await trainer.get_metrics("test_run")

        assert metrics["run_id"] == "test_run"
        assert metrics["total_epochs"] == 2
        assert len(metrics["jobs"]) == 2
        assert mock_adapter.status.call_count == 2

    async def test_log_progress(self, trainer, finetune_config, tmp_path):
        """Test logging progress to files."""
        # Mock components
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind and initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        metrics = {"avg_reward": 0.75, "min_reward": 0.5, "max_reward": 0.9, "num_trajectories": 10}

        output_dir = tmp_path / "logs"
        trainer.log_progress(epoch=0, metrics=metrics, output_dir=str(output_dir))

        # Check files were created
        assert (output_dir / "training_metrics.jsonl").exists()
        assert (output_dir / "reward_history.json").exists()

        # Verify metrics file content
        with open(output_dir / "training_metrics.jsonl") as f:
            log_entry = json.loads(f.readline())
            assert log_entry["epoch"] == 0
            assert log_entry["avg_reward"] == 0.75
            assert log_entry["num_trajectories"] == 10

        # Verify reward history
        with open(output_dir / "reward_history.json") as f:
            history = json.load(f)
            assert len(history) == 1
            assert history[0]["epoch"] == 0
            assert history[0]["avg_reward"] == 0.75

    def test_apply_curriculum_learning_disabled(self, trainer, finetune_config):
        """Test curriculum learning when disabled."""
        trainer.run_config = finetune_config
        trainer.curriculum_config = CurriculumLearningConfig(enabled=False)
        trainer._curriculum_state = {}

        # Create trajectories
        mock_trajectory = MagicMock(spec=Trajectory)
        collection = TrajectoryCollection(trajectories=[[mock_trajectory], [mock_trajectory]], run_id="test_run")

        # Apply curriculum (should return unchanged)
        filtered = trainer.apply_curriculum_learning(collection, epoch=0)

        assert filtered == collection

    async def test_apply_curriculum_learning_enabled(self, trainer, finetune_config):
        """Test curriculum learning when enabled."""
        finetune_config.curriculum_learning = CurriculumLearningConfig(
            enabled=True,
            initial_percentile=0.5,
            increment_percentile=0.25,
            expansion_interval=2,
            min_reward_diff=0.0  # Allow single-trajectory groups
        )

        # Mock components
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind and initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Create trajectories with different rewards
        traj1 = MagicMock(spec=Trajectory)
        traj1.reward = 0.9
        traj2 = MagicMock(spec=Trajectory)
        traj2.reward = 0.3
        traj3 = MagicMock(spec=Trajectory)
        traj3.reward = 0.5
        traj4 = MagicMock(spec=Trajectory)
        traj4.reward = 0.1

        collection = TrajectoryCollection(trajectories=[[traj1], [traj2], [traj3], [traj4]], run_id="test_run")

        # Apply curriculum at epoch 0 (50% percentile)
        filtered = trainer.apply_curriculum_learning(collection, epoch=0)

        # Should include top 50% (2 groups)
        assert len(filtered.trajectories) == 2
        assert trainer._curriculum_state["total_groups"] == 4
        assert len(trainer._curriculum_state["included_groups"]) == 2

    async def test_apply_curriculum_learning_expansion(self, trainer, finetune_config):
        """Test curriculum learning expansion at intervals."""
        finetune_config.curriculum_learning = CurriculumLearningConfig(
            enabled=True,
            initial_percentile=0.25,
            increment_percentile=0.25,
            expansion_interval=2,
            min_reward_diff=0.0  # Allow single-trajectory groups
        )

        # Mock components
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind and initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Manually set state as if we're at epoch 2
        trainer._curriculum_state = {
            "current_percentile": 0.25, "last_expansion_epoch": 0, "total_groups": 4, "included_groups": set([0])
        }

        # Create trajectories
        trajectories = []
        for i in range(4):
            traj = MagicMock(spec=Trajectory)
            traj.reward = (4 - i) * 0.25  # Decreasing rewards
            trajectories.append([traj])

        collection = TrajectoryCollection(trajectories=trajectories, run_id="test_run")

        # Apply at epoch 2 (should trigger expansion)
        filtered = trainer.apply_curriculum_learning(collection, epoch=2)

        # Should expand to 50%
        assert trainer._curriculum_state["current_percentile"] == 0.5
        assert trainer._curriculum_state["last_expansion_epoch"] == 2
        assert len(filtered.trajectories) == 2

    @patch('nat.plugins.openpipe.trainer.plt')
    def test_create_reward_plot(self, mock_plt, trainer, tmp_path):
        """Test creating reward visualization plots."""
        trainer.run_config = MagicMock()
        trainer.run_config.output_dir = tmp_path
        trainer.curriculum_config = CurriculumLearningConfig(enabled=False)
        trainer._curriculum_state = {"current_percentile": 1.0}

        # Add reward history
        trainer._reward_history = [{
            "epoch": 0, "avg_reward": 0.5, "min_reward": 0.3, "max_reward": 0.7
        }, {
            "epoch": 1, "avg_reward": 0.6, "min_reward": 0.4, "max_reward": 0.8
        }, {
            "epoch": 2, "avg_reward": 0.7, "min_reward": 0.5, "max_reward": 0.9
        }]

        # Mock matplotlib figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Create plot
        trainer._create_reward_plot(epoch=2, output_dir=tmp_path)

        # Verify plot was created
        mock_plt.subplots.assert_called_once()
        mock_ax.plot.assert_called()
        mock_ax.set_xlabel.assert_called_with('Epoch', fontsize=12)
        mock_ax.set_ylabel.assert_called_with('Reward', fontsize=12)
        mock_plt.savefig.assert_called_once()

    async def test_cleanup(self, trainer, finetune_config):
        """Test cleanup of resources."""
        # Create mock tasks
        eval_task = MagicMock()
        eval_task.done.return_value = False
        eval_task.cancel = MagicMock()

        training_task = MagicMock()
        training_task.done.return_value = False
        training_task.cancel = MagicMock()

        # Mock trajectory builder with tasks
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.evaluation_runs = {"run1": eval_task}

        # Mock trainer adapter with tasks
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()
        mock_adapter.training_jobs = {"job1": training_task}

        # Bind and initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Run cleanup
        await trainer.cleanup()

        # Verify tasks were cancelled
        eval_task.cancel.assert_called_once()
        training_task.cancel.assert_called_once()

    def test_curriculum_learning_single_group(self, trainer, finetune_config):
        """Test curriculum learning with single trajectory group."""
        finetune_config.curriculum_learning = CurriculumLearningConfig(enabled=True, random_subsample=0.5)
        trainer.run_config = finetune_config
        trainer.curriculum_config = finetune_config.curriculum_learning
        trainer._curriculum_state = {
            "current_percentile": 1.0, "last_expansion_epoch": -1, "total_groups": 0, "included_groups": set()
        }

        # Create single group with multiple trajectories
        trajectories = [MagicMock(spec=Trajectory) for _ in range(10)]
        for i, traj in enumerate(trajectories):
            traj.reward = i * 0.1

        collection = TrajectoryCollection(trajectories=[trajectories], run_id="test_run")

        # Apply curriculum
        with patch('random.sample', side_effect=lambda x, n: x[:n]):
            filtered = trainer.apply_curriculum_learning(collection, epoch=0)

        # Should subsample to 50%
        assert len(filtered.trajectories) == 1
        assert len(filtered.trajectories[0]) == 5

    async def test_curriculum_learning_no_variance(self, trainer, finetune_config):
        """Test curriculum learning filters groups with no variance."""
        finetune_config.curriculum_learning = CurriculumLearningConfig(enabled=True, min_reward_diff=0.01)

        # Mock components
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Bind and initialize
        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        # Create groups - one with no variance, one with variance
        traj1 = MagicMock(spec=Trajectory)
        traj1.reward = 0.5
        traj2 = MagicMock(spec=Trajectory)
        traj2.reward = 0.5  # Same reward - no variance

        traj3 = MagicMock(spec=Trajectory)
        traj3.reward = 0.3
        traj4 = MagicMock(spec=Trajectory)
        traj4.reward = 0.7  # Different rewards - has variance

        collection = TrajectoryCollection(trajectories=[[traj1, traj2], [traj3, traj4]], run_id="test_run")

        # Apply curriculum
        filtered = trainer.apply_curriculum_learning(collection, epoch=0)

        # Should filter out group with no variance, keep the one with variance
        assert len(filtered.trajectories) == 1
        assert filtered.trajectories[0] == [traj3, traj4]
