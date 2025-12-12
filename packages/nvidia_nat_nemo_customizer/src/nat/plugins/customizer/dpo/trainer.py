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
"""
NeMo Customizer Trainer for DPO finetuning.

This module provides a Trainer implementation that orchestrates data collection
via trajectory builders and submits training jobs to NeMo Customizer.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.finetuning.interfaces.finetuning_runner import Trainer

from .config import NeMoCustomizerTrainerConfig

logger = logging.getLogger(__name__)


class NeMoCustomizerTrainer(Trainer):
    """
    Trainer for NeMo Customizer DPO/SFT finetuning.

    Unlike epoch-based trainers, this trainer:
    1. Runs the trajectory builder multiple times (num_runs) to collect data
    2. Aggregates all trajectories into a single dataset
    3. Submits the dataset to NeMo Customizer for training
    4. Monitors the training job until completion

    The actual training epochs are handled by NeMo Customizer via hyperparameters.
    """

    def __init__(self, trainer_config: NeMoCustomizerTrainerConfig, **kwargs) -> None:
        """
        Initialize the NeMo Customizer Trainer.

        Args:
            trainer_config: Configuration for the trainer
        """
        super().__init__(trainer_config)

        self.trainer_config: NeMoCustomizerTrainerConfig = trainer_config

        # Track job references and metrics
        self._job_ref: TrainingJobRef | None = None
        self._run_id: str | None = None

        # Track collected data across runs
        self._all_trajectories: list[list[Trajectory]] = []
        self._run_metrics: list[dict[str, Any]] = []

        # Progress tracking
        self._collection_history: list[dict[str, Any]] = []

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """
        Initialize the trainer and its components.

        Note: Curriculum learning is not supported for DPO training.
        """
        logger.info("Initializing NeMo Customizer Trainer")

        # Store run config but skip curriculum learning setup
        self.run_config = run_config
        self.trainer_config.reward = self.run_config.reward_function

        # Disable curriculum learning for DPO
        self.curriculum_config = None
        self._curriculum_state = {
            "current_percentile": 1.0,
            "last_expansion_epoch": -1,
            "total_groups": 0,
            "included_groups": set(),
        }

        # Initialize components
        await self.trajectory_builder.initialize(run_config)
        await self.trainer_adapter.initialize(run_config)

        # Generate unique run ID
        self._run_id = f"nemo_dpo_{uuid.uuid4().hex[:8]}"

        logger.info(f"NeMo Customizer Trainer initialized with run ID: {self._run_id}")

    async def run_epoch(self, epoch: int, run_id: str) -> TrainingJobRef | None:
        """
        Run a single data collection run.

        For NeMo Customizer, this collects trajectories without submitting
        to training. The actual submission happens in run().

        Args:
            epoch: The current run number (0-indexed)
            run_id: Unique identifier for this training run

        Returns:
            None (trajectories are accumulated, not submitted per-run)
        """
        logger.info(f"Starting data collection run {epoch + 1}/{self.trainer_config.num_runs}")

        run_meta = {
            "run_number": epoch,
            "run_id": run_id,
            "trainer_config": self.trainer_config.model_dump(),
        }

        # Run trajectory builder
        await self.trajectory_builder.start_run(run_id=f"{run_id}_run{epoch}", meta=run_meta)

        # Finalize and get trajectories
        trajectory_collection = await self.trajectory_builder.finalize(run_id=f"{run_id}_run{epoch}", meta=run_meta)

        if not trajectory_collection.trajectories:
            logger.warning(f"No trajectories collected for run {epoch}")
            return None

        # Calculate metrics for this run
        run_rewards = []
        num_trajectories = 0
        num_dpo_pairs = 0

        for trajectory_group in trajectory_collection.trajectories:
            for trajectory in trajectory_group:
                num_trajectories += 1
                run_rewards.append(trajectory.reward)
                # Count DPO pairs (each trajectory has one DPOItem)
                num_dpo_pairs += len(trajectory.episode)

        metrics = {
            "run_number": epoch,
            "num_trajectories": num_trajectories,
            "num_dpo_pairs": num_dpo_pairs,
            "avg_reward": sum(run_rewards) / len(run_rewards) if run_rewards else 0.0,
            "min_reward": min(run_rewards) if run_rewards else 0.0,
            "max_reward": max(run_rewards) if run_rewards else 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        self._run_metrics.append(metrics)

        logger.info(f"Run {epoch + 1}: Collected {num_trajectories} trajectories, "
                    f"{num_dpo_pairs} DPO pairs, avg reward: {metrics['avg_reward']:.4f}")

        # Accumulate trajectories
        self._all_trajectories.extend(trajectory_collection.trajectories)

        # Log progress
        self.log_progress(epoch, metrics)

        return None  # No job submitted per-run

    async def run(self, num_epochs: int) -> list[TrainingJobStatus]:
        """
        Run the complete DPO data collection and training workflow.

        Args:
            num_epochs: Ignored for NeMo Customizer (uses trainer_config.num_runs)

        Returns:
            list[TrainingJobStatus]: Status of the training job
        """
        if not self._run_id:
            raise RuntimeError("Trainer not initialized. Call initialize() first.")

        num_runs = self.trainer_config.num_runs
        logger.info(f"Starting NeMo Customizer DPO workflow with {num_runs} data collection runs")

        # Phase 1: Collect data from multiple runs
        for run_idx in range(num_runs):
            try:
                await self.run_epoch(run_idx, self._run_id)
            except Exception as e:
                logger.error(f"Error during data collection run {run_idx}: {e}")
                if not self.trainer_config.continue_on_collection_error:
                    return [
                        TrainingJobStatus(
                            run_id=self._run_id,
                            backend="nemo-customizer",
                            status=TrainingStatusEnum.FAILED,
                            message=f"Data collection failed at run {run_idx}: {e}",
                            metadata={"run_number": run_idx},
                        )
                    ]

        # Check if we have any data
        if not self._all_trajectories:
            logger.error("No trajectories collected from any run")
            return [
                TrainingJobStatus(
                    run_id=self._run_id,
                    backend="nemo-customizer",
                    status=TrainingStatusEnum.FAILED,
                    message="No trajectories collected",
                )
            ]

        # Calculate total statistics
        total_trajectories = len(self._all_trajectories)
        total_dpo_pairs = sum(
            len(traj.episode) for group in self._all_trajectories
            for traj in (group if isinstance(group, list) else [group]))

        logger.info(f"Data collection complete: {total_trajectories} trajectory groups, "
                    f"~{total_dpo_pairs} total DPO pairs from {num_runs} runs")

        # Phase 2: Submit aggregated trajectories for training
        try:
            trajectory_collection = TrajectoryCollection(
                trajectories=self._all_trajectories,
                run_id=self._run_id,
            )

            # Apply deduplication if configured
            if self.trainer_config.deduplicate_pairs:
                trajectory_collection = self._deduplicate_trajectories(trajectory_collection)

            # Apply sampling if configured
            if self.trainer_config.max_pairs is not None:
                trajectory_collection = self._sample_trajectories(trajectory_collection, self.trainer_config.max_pairs)

            self._job_ref = await self.trainer_adapter.submit(trajectory_collection)

            logger.info(f"Submitted training job: {self._job_ref.metadata.get('job_id')}")

        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            return [
                TrainingJobStatus(
                    run_id=self._run_id,
                    backend="nemo-customizer",
                    status=TrainingStatusEnum.FAILED,
                    message=f"Failed to submit training job: {e}",
                )
            ]

        # Phase 3: Wait for training completion
        if self.trainer_config.wait_for_completion:
            logger.info("Waiting for training job to complete...")
            final_status = await self.trainer_adapter.wait_until_complete(self._job_ref)

            # Log final metrics
            self._log_final_metrics(final_status)

            return [final_status]
        else:
            # Return immediately with pending status
            return [
                TrainingJobStatus(
                    run_id=self._run_id,
                    backend="nemo-customizer",
                    status=TrainingStatusEnum.RUNNING,
                    message="Training job submitted (not waiting for completion)",
                    metadata=self._job_ref.metadata,
                )
            ]

    def _deduplicate_trajectories(self, collection: TrajectoryCollection) -> TrajectoryCollection:
        """Remove duplicate DPO pairs based on prompt+responses."""
        seen = set()
        unique_groups = []

        for group in collection.trajectories:
            unique_trajectories = []
            for traj in group:
                for item in traj.episode:
                    # Create a hashable key from prompt and responses
                    prompt_str = (str(item.prompt) if hasattr(item, "prompt") else "")
                    key = (
                        prompt_str,
                        getattr(item, "chosen_response", ""),
                        getattr(item, "rejected_response", ""),
                    )
                    if key not in seen:
                        seen.add(key)
                        unique_trajectories.append(traj)
                        break  # Only add trajectory once

            if unique_trajectories:
                unique_groups.append(unique_trajectories)

        original_count = sum(len(g) for g in collection.trajectories)
        new_count = sum(len(g) for g in unique_groups)
        logger.info(f"Deduplication: {original_count} -> {new_count} trajectories")

        return TrajectoryCollection(trajectories=unique_groups, run_id=collection.run_id)

    def _sample_trajectories(self, collection: TrajectoryCollection, max_pairs: int) -> TrajectoryCollection:
        """Sample trajectories to limit dataset size."""
        import random

        all_trajectories = []
        for group in collection.trajectories:
            all_trajectories.extend(group)

        if len(all_trajectories) <= max_pairs:
            return collection

        # Sample randomly
        sampled = random.sample(all_trajectories, max_pairs)
        logger.info(f"Sampling: {len(all_trajectories)} -> {max_pairs} trajectories")

        return TrajectoryCollection(
            trajectories=[[t] for t in sampled],
            run_id=collection.run_id,
        )

    async def get_metrics(self, run_id: str) -> dict[str, Any]:
        """Get training metrics for the run."""
        metrics = {
            "run_id": run_id,
            "num_collection_runs": len(self._run_metrics),
            "collection_runs": self._run_metrics,
            "total_trajectory_groups": len(self._all_trajectories),
        }

        if self._job_ref:
            try:
                status = await self.trainer_adapter.status(self._job_ref)
                metrics["training_job"] = {
                    "job_id": self._job_ref.metadata.get("job_id"),
                    "status": status.status.value,
                    "progress": status.progress,
                    "message": status.message,
                }
            except Exception as e:
                metrics["training_job"] = {"error": str(e)}

        return metrics

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up NeMo Customizer Trainer resources")

        # Cancel any running trajectory builder tasks
        if hasattr(self.trajectory_builder, "evaluation_runs"):
            for run_id, task in self.trajectory_builder.evaluation_runs.items():
                if not task.done():
                    logger.info(f"Cancelling evaluation task for run {run_id}")
                    task.cancel()

        # Clear accumulated data
        self._all_trajectories.clear()
        self._run_metrics.clear()

        logger.info("NeMo Customizer Trainer cleanup completed")

    def log_progress(self, epoch: int, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log data collection progress."""
        out_dir = Path(output_dir) if output_dir else self.run_config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Store in history
        progress_entry = {
            "run_number": epoch,
            "timestamp": datetime.now().isoformat(),
            "run_id": self._run_id,
            **metrics,
        }
        self._collection_history.append(progress_entry)

        # Log to JSON file
        log_file = out_dir / "data_collection_progress.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(progress_entry) + "\n")

        # Save collection history
        history_file = out_dir / "collection_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self._collection_history, f, indent=2)

        logger.info(f"Run {epoch + 1}: {metrics.get('num_dpo_pairs', 0)} DPO pairs, "
                    f"avg reward: {metrics.get('avg_reward', 0):.4f}")

    def _log_final_metrics(self, final_status: TrainingJobStatus) -> None:
        """Log final training metrics."""
        out_dir = self.run_config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        final_metrics = {
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            "status": final_status.status.value,
            "message": final_status.message,
            "progress": final_status.progress,
            "num_collection_runs": len(self._run_metrics),
            "total_trajectory_groups": len(self._all_trajectories),
            "collection_summary": {
                "total_trajectories":
                    sum(m.get("num_trajectories", 0) for m in self._run_metrics),
                "total_dpo_pairs":
                    sum(m.get("num_dpo_pairs", 0) for m in self._run_metrics),
                "avg_reward": (sum(m.get("avg_reward", 0)
                                   for m in self._run_metrics) / len(self._run_metrics) if self._run_metrics else 0.0),
            },
            "job_metadata": self._job_ref.metadata if self._job_ref else None,
        }

        # Save final metrics
        metrics_file = out_dir / "final_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Training completed with status: {final_status.status.value}")
