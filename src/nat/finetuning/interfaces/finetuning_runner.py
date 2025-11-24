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
import math

from nat.data_models.finetuning import (
    CurriculumLearningConfig,
    RLTrainerConfig,
    TrainerRunConfig,
    Trajectory,
    TrajectoryCollection,
    TrainingJobRef,
    TrainingJobStatus
)
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)


class FinetuningRunner(ABC):
    """
    Abstract interface for running finetuning workflows.

    The FinetuningRunner orchestrates the entire finetuning process by:
    1. Running evaluations to generate trajectories via TrajectoryBuilder
    2. Submitting trajectories for training via TrainerAdapter
    3. Managing multiple epochs of training
    """

    def __init__(
            self,
            trainer_config: RLTrainerConfig,
            run_config: TrainerRunConfig,
            backend: str,
            **kwargs
    ) -> None:
        """
        Initialize the FinetuningRunner.

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

        # Curriculum learning state
        self._curriculum_state = {
            "current_percentile": self.curriculum_config.initial_percentile,
            "last_expansion_epoch": -1,
            "total_groups": 0,
            "included_groups": set()
        }

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

        config = self.run_config.validation_config_file if self.run_config.validation_config_file else self.run_config.config_file

        # Create a temporary run config with validation dataset
        validation_run_config = TrainerRunConfig(
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
    def _create_trajectory_builder(self, run_config: TrainerRunConfig):
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

        This method:
        1. Sorts trajectory groups by average reward (difficulty)
        2. Filters out groups with no reward variance (no learning signal)
        3. Selects appropriate groups based on curriculum progression
        4. Expands curriculum at specified intervals

        Args:
            trajectory_collection: The complete collection of trajectories
            epoch: Current epoch number

        Returns:
            TrajectoryCollection: Filtered trajectories for training
        """
        if not self.curriculum_config.enabled:
            # Curriculum learning disabled, return all trajectories
            return trajectory_collection

        if len(trajectory_collection.trajectories) == 1:
            # Only one group, so we pick only run a random subsample if specified
            if self.curriculum_config.random_subsample is not None:
                import random
                fraction = self.curriculum_config.random_subsample
                trajectory_group = trajectory_collection.trajectories[0]
                max_required_trajectories = int(math.ceil(len(trajectory_group) * fraction))
                if len(trajectory_group) > max_required_trajectories:
                    selected_trajectories = random.sample(
                        trajectory_group,
                        max_required_trajectories
                    )
                    logger.info(
                        "After random subsampling %.2f, using %d trajectories from single group",
                        fraction,
                        len(selected_trajectories)
                    )
                    return TrajectoryCollection(
                        trajectories=[selected_trajectories],
                        run_id=trajectory_collection.run_id
                    )

            return trajectory_collection

        # Calculate statistics for each trajectory group
        group_stats = []
        for group_idx, trajectory_group in enumerate(trajectory_collection.trajectories):
            if not trajectory_group:
                continue

            rewards = [t.reward for t in trajectory_group]
            avg_reward = sum(rewards) / len(rewards)
            variance = sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)
            max_diff = max(rewards) - min(rewards)

            # Skip groups with insufficient reward variance (no learning signal)
            if max_diff < self.curriculum_config.min_reward_diff:
                logger.info(
                    "Skipping trajectory group %d with max_diff %.6f < %.6f (no learning signal)",
                    group_idx, max_diff, self.curriculum_config.min_reward_diff
                )
                continue

            group_stats.append({
                "index": group_idx,
                "avg_reward": avg_reward,
                "variance": variance,
                "trajectories": trajectory_group
            })

        if not group_stats:
            logger.warning("No trajectory groups with sufficient variance found")
            return TrajectoryCollection(
                trajectories=[],
                run_id=trajectory_collection.run_id
            )

        # Sort groups by average reward (difficulty)
        group_stats.sort(
            key=lambda x: x["avg_reward"],
            reverse=not self.curriculum_config.sort_ascending
        )

        # Store total groups if first epoch
        if self._curriculum_state["total_groups"] == 0:
            self._curriculum_state["total_groups"] = len(group_stats)

        # Check if we should expand the curriculum
        epochs_since_expansion = epoch - self._curriculum_state["last_expansion_epoch"]
        should_expand = (
            epochs_since_expansion >= self.curriculum_config.expansion_interval
            and self._curriculum_state["current_percentile"] < 1.0
        )

        if should_expand:
            # Expand curriculum by increment_percentile
            old_percentile = self._curriculum_state["current_percentile"]
            self._curriculum_state["current_percentile"] = min(
                1.0,
                old_percentile + self.curriculum_config.increment_percentile
            )
            self._curriculum_state["last_expansion_epoch"] = epoch

            logger.info(
                "Expanding curriculum at epoch %d: %.1f%% -> %.1f%% of trajectory groups",
                epoch,
                old_percentile * 100,
                self._curriculum_state["current_percentile"] * 100
            )

        # Calculate number of groups to include
        num_groups_to_include = max(
            1,  # Always include at least one group
            int(math.ceil(len(group_stats) * self._curriculum_state["current_percentile"]))
        )

        # Select the appropriate groups
        selected_groups = group_stats[:num_groups_to_include]

        # Track which groups are included
        included_indices = {g["index"] for g in selected_groups}
        new_groups = included_indices - self._curriculum_state["included_groups"]
        if new_groups:
            logger.info(
                "Adding %d new trajectory groups to curriculum at epoch %d",
                len(new_groups), epoch
            )
        self._curriculum_state["included_groups"] = included_indices

        # Log curriculum statistics
        selected_trajectories = [g["trajectories"] for g in selected_groups]
        total_trajectories = sum(len(traj_list) for traj_list in selected_trajectories)

        logger.info(
            "Curriculum learning at epoch %d: Using %d/%d groups (%.1f%%), "
            "%d total trajectories. Avg reward range: [%.4f, %.4f]",
            epoch,
            len(selected_groups),
            len(group_stats),
            self._curriculum_state["current_percentile"] * 100,
            total_trajectories,
            selected_groups[-1]["avg_reward"] if selected_groups else 0,
            selected_groups[0]["avg_reward"] if selected_groups else 0
        )

        if self.curriculum_config.random_subsample is not None:
            # Randomly select only a fraction of trajectory groups to use
            import random
            fraction = self.curriculum_config.random_subsample
            # Max required groups is the theoretical max based on fraction
            max_required_groups = int(math.ceil(len(group_stats) * fraction))
            # Now select at most that many groups from selected groups
            if len(selected_groups) > max_required_groups:
                selected_groups = random.sample(
                    selected_groups,
                    max_required_groups
                )
                # Rebuild selected trajectories
                selected_trajectories = [g["trajectories"] for g in selected_groups]
                logger.info(
                    "After random subsampling %.2f, using %d trajectory groups",
                    fraction,
                    len(selected_groups)
                )

        return TrajectoryCollection(
            trajectories=selected_trajectories,
            run_id=trajectory_collection.run_id
        )

    def get_curriculum_state(self) -> dict[str, Any]:
        """
        Get the current state of curriculum learning.

        Returns:
            dict: Current curriculum state including percentile and group statistics
        """
        # Convert set to list for JSON serialization
        state = {
            "current_percentile": self._curriculum_state["current_percentile"],
            "last_expansion_epoch": self._curriculum_state["last_expansion_epoch"],
            "total_groups": self._curriculum_state["total_groups"],
            "included_groups": list(self._curriculum_state["included_groups"]),
            "config": self.curriculum_config.model_dump() if self.curriculum_config else None
        }
        return state
