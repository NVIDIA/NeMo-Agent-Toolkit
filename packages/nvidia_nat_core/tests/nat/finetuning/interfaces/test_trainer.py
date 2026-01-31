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

import pytest

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import TrajectoryCollection
from nat.eval.config import EvaluationRunOutput
from nat.finetuning.interfaces.finetuning_runner import Trainer
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder


class ConcreteTrainer(Trainer):
    """Concrete implementation of Trainer for testing."""

    def __init__(self, trainer_config: TrainerConfig, **kwargs):
        super().__init__(trainer_config, **kwargs)
        self.epochs_run = []
        self.logged_progress = []

    async def run_epoch(self, epoch: int, run_id: str) -> TrainingJobRef:
        """Run a single epoch of training."""
        self.epochs_run.append((epoch, run_id))
        return TrainingJobRef(run_id=run_id, backend="test_backend", metadata={"epoch": epoch})

    async def run(self, num_epochs: int) -> list[TrainingJobStatus]:
        """Run the complete finetuning workflow."""
        statuses = []
        for epoch in range(num_epochs):
            run_id = f"run_{epoch}"
            await self.run_epoch(epoch, run_id)
            statuses.append(
                TrainingJobStatus(run_id=run_id,
                                  backend="test_backend",
                                  status=TrainingStatusEnum.COMPLETED,
                                  progress=100.0,
                                  message=f"Epoch {epoch} completed"))
        return statuses

    async def get_metrics(self, run_id: str) -> dict[str, Any]:
        """Get training metrics for a specific run."""
        return {"run_id": run_id, "loss": 0.5, "accuracy": 0.95}

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def log_progress(self, epoch: int, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log training progress."""
        self.logged_progress.append({"epoch": epoch, "metrics": metrics, "output_dir": output_dir})


class TestTrainer:
    """Tests for the Trainer interface."""

    @pytest.fixture
    def trainer_config(self):
        """Create a test trainer config."""

        # Create a concrete config class
        class TestTrainerConfig(TrainerConfig, name="test_trainer_with_reward"):
            pass

        return TestTrainerConfig(reward=RewardFunctionConfig(name="test_reward"))

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create a test finetune config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{\"input\": \"test\"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        return FinetuneConfig(run_configuration=run_config, curriculum_learning=CurriculumLearningConfig())

    @pytest.fixture
    def trainer(self, trainer_config):
        """Create a concrete trainer instance."""
        return ConcreteTrainer(trainer_config=trainer_config)

    async def test_trainer_initialization(self, trainer, trainer_config):
        """Test that trainer initializes with correct configuration."""
        assert trainer.trainer_config == trainer_config
        assert trainer.run_config is None
        assert trainer.curriculum_config is None
        assert trainer.trajectory_builder is None
        assert trainer.trainer_adapter is None

    async def test_trainer_initialize(self, trainer, finetune_config):
        """Test trainer initialization."""
        # Mock the components
        mock_trajectory_builder = MagicMock(spec=TrajectoryBuilder)
        mock_trajectory_builder.initialize = AsyncMock()
        mock_trainer_adapter = MagicMock(spec=TrainerAdapter)
        mock_trainer_adapter.initialize = AsyncMock()

        trainer.trajectory_builder = mock_trajectory_builder
        trainer.trainer_adapter = mock_trainer_adapter

        await trainer.initialize(finetune_config)

        assert trainer.run_config == finetune_config
        assert trainer.curriculum_config == finetune_config.curriculum_learning
        mock_trajectory_builder.initialize.assert_called_once_with(finetune_config)
        mock_trainer_adapter.initialize.assert_called_once_with(finetune_config)

    async def test_trainer_bind_components(self, trainer):
        """Test binding trajectory builder and trainer adapter."""
        mock_trajectory_builder = MagicMock(spec=TrajectoryBuilder)
        mock_trainer_adapter = MagicMock(spec=TrainerAdapter)

        await trainer.bind_components(mock_trajectory_builder, mock_trainer_adapter)

        assert trainer.trajectory_builder == mock_trajectory_builder
        assert trainer.trainer_adapter == mock_trainer_adapter

    async def test_trainer_run_epoch(self, trainer):
        """Test running a single epoch."""
        job_ref = await trainer.run_epoch(epoch=0, run_id="test_run")

        assert isinstance(job_ref, TrainingJobRef)
        assert job_ref.run_id == "test_run"
        assert job_ref.backend == "test_backend"
        assert job_ref.metadata["epoch"] == 0
        assert (0, "test_run") in trainer.epochs_run

    async def test_trainer_run(self, trainer):
        """Test running multiple epochs."""
        statuses = await trainer.run(num_epochs=3)

        assert len(statuses) == 3
        assert all(isinstance(status, TrainingJobStatus) for status in statuses)
        assert all(status.status == TrainingStatusEnum.COMPLETED for status in statuses)
        assert len(trainer.epochs_run) == 3

    async def test_trainer_get_metrics(self, trainer):
        """Test getting metrics for a run."""
        metrics = await trainer.get_metrics("test_run")

        assert isinstance(metrics, dict)
        assert "run_id" in metrics
        assert metrics["run_id"] == "test_run"

    async def test_trainer_cleanup(self, trainer):
        """Test cleanup method."""
        await trainer.cleanup()
        # Just verify it doesn't raise

    def test_trainer_log_progress(self, trainer):
        """Test logging progress."""
        metrics = {"loss": 0.5, "accuracy": 0.95}
        trainer.log_progress(epoch=1, metrics=metrics, output_dir="/tmp/logs")

        assert len(trainer.logged_progress) == 1
        assert trainer.logged_progress[0]["epoch"] == 1
        assert trainer.logged_progress[0]["metrics"] == metrics
        assert trainer.logged_progress[0]["output_dir"] == "/tmp/logs"

    async def test_trainer_run_validation_evaluation(self, trainer, finetune_config, tmp_path):
        """Test running validation evaluation."""
        validation_dataset = tmp_path / "validation.jsonl"
        validation_dataset.write_text('{\"input\": \"test\"}')

        # Set validation dataset in config
        finetune_config.run_configuration.validation_dataset = str(validation_dataset)

        # Mock the evaluation output
        mock_eval_output = MagicMock(spec=EvaluationRunOutput)
        mock_metric = MagicMock()
        mock_metric.score = 0.8
        mock_eval_output.evaluation_results = [("test_reward", MagicMock(eval_output_items=[mock_metric, mock_metric]))]

        # Mock trajectory builder
        mock_trajectory_builder = MagicMock(spec=TrajectoryBuilder)
        mock_trajectory_builder.run_eval = AsyncMock(return_value=mock_eval_output)
        mock_trajectory_builder.run_config = MagicMock()
        mock_trajectory_builder.run_config.run_configuration = finetune_config.run_configuration

        trainer.trajectory_builder = mock_trajectory_builder
        trainer.run_config = finetune_config

        metrics = await trainer.run_validation_evaluation(epoch=1, run_id="test_run")

        assert "epoch" in metrics
        assert metrics["epoch"] == 1
        assert "dataset_type" in metrics
        assert metrics["dataset_type"] == "validation"
        assert "avg_reward" in metrics
        assert metrics["avg_reward"] == 0.8

    async def test_trainer_run_validation_evaluation_error_handling(self, trainer, finetune_config, tmp_path):
        """Test validation evaluation error handling."""
        validation_dataset = tmp_path / "validation.jsonl"
        validation_dataset.write_text('{\"input\": \"test\"}')

        # Set validation dataset in config
        finetune_config.run_configuration.validation_dataset = str(validation_dataset)

        # Mock trajectory builder to raise error
        mock_trajectory_builder = MagicMock(spec=TrajectoryBuilder)
        mock_trajectory_builder.run_eval = AsyncMock(side_effect=Exception("Test error"))
        mock_trajectory_builder.run_config = MagicMock()
        mock_trajectory_builder.run_config.run_configuration = finetune_config.run_configuration

        trainer.trajectory_builder = mock_trajectory_builder
        trainer.run_config = finetune_config

        metrics = await trainer.run_validation_evaluation(epoch=1, run_id="test_run")

        assert "error" in metrics
        assert metrics["error"] == "Test error"
        assert metrics["avg_reward"] == 0.0

    def test_trainer_calculate_validation_metrics(self, trainer, trainer_config):
        """Test calculating validation metrics from evaluation output."""
        # Set the trainer config with reward
        trainer.trainer_config = trainer_config

        mock_eval_output = MagicMock(spec=EvaluationRunOutput)
        mock_metric1 = MagicMock()
        mock_metric1.score = 0.8
        mock_metric2 = MagicMock()
        mock_metric2.score = 0.6
        mock_eval_output.evaluation_results = [("test_reward",
                                                MagicMock(eval_output_items=[mock_metric1, mock_metric2]))]

        metrics = trainer._calculate_validation_metrics(mock_eval_output)

        assert metrics["avg_reward"] == 0.7
        assert metrics["min_reward"] == 0.6
        assert metrics["max_reward"] == 0.8
        assert metrics["num_examples"] == 2

    def test_trainer_calculate_validation_metrics_no_rewards(self, trainer, trainer_config):
        """Test calculating validation metrics with no rewards."""
        trainer.trainer_config = trainer_config

        mock_eval_output = MagicMock(spec=EvaluationRunOutput)
        mock_eval_output.evaluation_results = []

        metrics = trainer._calculate_validation_metrics(mock_eval_output)

        assert metrics["avg_reward"] == 0.0
        assert metrics["num_examples"] == 0

    async def test_trainer_apply_curriculum_learning_not_implemented(self, trainer):
        """Test that apply_curriculum_learning raises NotImplementedError by default."""
        mock_trajectory_collection = MagicMock(spec=TrajectoryCollection)

        with pytest.raises(NotImplementedError, match="Curriculum learning not implemented"):
            trainer.apply_curriculum_learning(mock_trajectory_collection, epoch=1)

    async def test_trainer_curriculum_state_initialization(self, trainer, finetune_config):
        """Test that curriculum state is properly initialized."""
        # Mock components
        mock_trajectory_builder = MagicMock(spec=TrajectoryBuilder)
        mock_trajectory_builder.initialize = AsyncMock()
        mock_trainer_adapter = MagicMock(spec=TrainerAdapter)
        mock_trainer_adapter.initialize = AsyncMock()

        trainer.trajectory_builder = mock_trajectory_builder
        trainer.trainer_adapter = mock_trainer_adapter

        await trainer.initialize(finetune_config)

        assert "_curriculum_state" in trainer.__dict__
        assert trainer._curriculum_state["current_percentile"] == trainer.curriculum_config.initial_percentile
        assert trainer._curriculum_state["last_expansion_epoch"] == -1
        assert trainer._curriculum_state["total_groups"] == 0
        assert isinstance(trainer._curriculum_state["included_groups"], set)

    async def test_trainer_config_reward_field(self, trainer_config):
        """Test that TrainerConfig has reward field properly set."""
        assert trainer_config.reward is not None
        assert isinstance(trainer_config.reward, RewardFunctionConfig)
        assert trainer_config.reward.name == "test_reward"

    async def test_trainer_config_reward_field_default(self):
        """Test that TrainerConfig reward field defaults to None."""

        class TestTrainerConfigNoReward(TrainerConfig, name="test_trainer_no_reward"):
            pass

        config = TestTrainerConfigNoReward()
        assert config.reward is None
