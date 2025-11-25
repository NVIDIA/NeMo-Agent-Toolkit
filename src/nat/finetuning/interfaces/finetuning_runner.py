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

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from nat.data_models.finetuning import (
    FinetuneRunConfig,
    TrainerConfig,
    TrajectoryCollection,
    TrainingJobRef,
    TrainingJobStatus,
)
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder
from nat.eval.config import EvaluationRunOutput

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    Abstract interface for running finetuning workflows.

    The Trainer orchestrates the entire finetuning process by:
    1. Running evaluations to generate trajectories via TrajectoryBuilder
    2. Submitting trajectories for training via TrainerAdapter
    3. Managing multiple epochs of training
    """

    def __init__(
            self,
            trainer_config: TrainerConfig,
            run_config: FinetuneRunConfig,
            backend: str,
            **kwargs
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            trainer_config: Configuration for the trainer backend
            run_config: Configuration for the training run
            backend: Backend identifier
            curriculum_config: Optional curriculum learning configuration
        """
        self.trainer_config = trainer_config
        self.run_config = run_config
        self._backend = backend
        self.curriculum_config = run_config.curriculum_learning
        self.trajectory_builder: TrajectoryBuilder | None = None
        self.trainer_adapter: TrainerAdapter | None = None

        # Curriculum learning state
        self._curriculum_state = {
            "current_percentile": self.curriculum_config.initial_percentile,
            "last_expansion_epoch": -1,
            "total_groups": 0,
            "included_groups": set()
        }

    async def bind_components(self,
        trajectory_builder: TrajectoryBuilder,
        trainer_adapter: TrainerAdapter
    ) -> None:
        """
        Bind the TrajectoryBuilder and TrainerAdapter components.

        Args:
            trajectory_builder: Instance of TrajectoryBuilder
            trainer_adapter: Instance of TrainerAdapter
        """
        self.trajectory_builder = trajectory_builder
        self.trainer_adapter = trainer_adapter

    @property
    def backend(self) -> str:
        """Return the backend type."""
        return self._backend

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the runner and its components.

        This should:
        - Initialize the TrajectoryBuilder
        - Initialize the TrainerAdapter
        - Verify connectivity to backend services
        """
        raise NotImplementedError

    @abstractmethod
    async def run_epoch(self, epoch: int, run_id: str) -> TrainingJobRef:
        """
        Run a single epoch of training.

        Args:
            epoch: The current epoch number (0-indexed)
            run_id: Unique identifier for this training run

        Returns:
            TrainingJobRef: Reference to the submitted training job
        """
        raise NotImplementedError

    @abstractmethod
    async def run(self, num_epochs: int) -> list[TrainingJobStatus]:
        """
        Run the complete finetuning workflow for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            list[TrainingJobStatus]: Status of all training jobs
        """
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self, run_id: str) -> dict[str, Any]:
        """
        Get training metrics for a specific run.

        Args:
            run_id: The run identifier

        Returns:
            dict: Metrics from the training run
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up any resources used by the runner.
        """
        raise NotImplementedError

    @abstractmethod
    def log_progress(
            self,
            epoch: int,
            metrics: dict[str, Any],
            output_dir: str | None = None
    ) -> None:
        """
        Log training progress for monitoring.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
            output_dir: Optional output directory override
        """
        raise NotImplementedError

    async def run_validation_evaluation(
            self,
            epoch: int,
            run_id: str,
            validation_dataset: str | Path
    ) -> dict[str, Any]:
        """
        Run evaluation on validation dataset to collect rewards.

        This method creates a temporary TrainerRunConfig with the validation
        dataset and runs evaluation to collect rewards without training.

        Args:
            epoch: Current epoch number
            run_id: Unique identifier for this training run
            validation_dataset: Path to the validation dataset

        Returns:
            dict: Validation metrics including average reward
        """
        logger.info("Running validation evaluation for epoch %d", epoch+1)

        config = self.run_config.validation_config_file if (
            self.run_config.validation_config_file) else self.run_config.config_file

        # Create a temporary run config with validation dataset
        validation_run_config = FinetuneRunConfig(
            config_file=config,
            target_functions=self.run_config.target_functions,
            dataset=validation_dataset,
            result_json_path=self.run_config.result_json_path,
            endpoint=self.run_config.endpoint,
            endpoint_timeout=self.run_config.endpoint_timeout,
            override=self.run_config.override
        )

        # Create a temporary trajectory builder for validation
        validation_builder = self._create_trajectory_builder(
            validation_run_config
        )

        try:
            # Run evaluation
            eval_output = await validation_builder.run_eval()

            # Calculate validation metrics from eval output
            validation_metrics = self._calculate_validation_metrics(
                eval_output
            )
            validation_metrics["epoch"] = epoch
            validation_metrics["dataset_type"] = "validation"

            logger.info(
                "Validation metrics for epoch %d: %s",
                epoch,
                validation_metrics
            )
            return validation_metrics

        except Exception as e:
            logger.error("Error during validation evaluation: %s", e)
            return {
                "epoch": epoch,
                "dataset_type": "validation",
                "error": str(e),
                "avg_reward": 0.0,
                "num_examples": 0
            }

    @abstractmethod
    def _create_trajectory_builder(self, run_config: FinetuneRunConfig):
        """
        Create a trajectory builder instance for the specific backend.

        Args:
            run_config: Configuration for the run

        Returns:
            TrajectoryBuilder: Instance of trajectory builder
        """
        raise NotImplementedError

    def _calculate_validation_metrics(self, eval_output:EvaluationRunOutput
                                      ) -> dict[str, Any]:
        """
        Calculate validation metrics from evaluation output.

        Args:
            eval_output: Output from evaluation run

        Returns:
            dict: Calculated metrics
        """
        # Default implementation - subclasses can override for
        # backend-specific metrics
        metrics = {
            "avg_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "num_examples": 0
        }

        rewards = []
        for metric_name, metric_value in eval_output.evaluation_results:
            if metric_name == self.trainer_config.reward.name:
                reward_results = metric_value.eval_output_items
                for reward_item in reward_results:
                    rewards.append(reward_item.score)


        if rewards:
            metrics["avg_reward"] = sum(rewards) / len(rewards)
            metrics["min_reward"] = min(rewards)
            metrics["max_reward"] = max(rewards)
            metrics["num_examples"] = len(rewards)

        return metrics

    def apply_curriculum_learning(
        self,
        trajectory_collection: TrajectoryCollection,
        epoch: int
    ) -> TrajectoryCollection:
        """
        Apply curriculum learning to filter trajectory groups based on difficulty.
        """
        raise NotImplementedError("Curriculum learning not implemented for this backend.")
