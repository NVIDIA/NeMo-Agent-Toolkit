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

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.finetuning.interfaces.finetuning_runner import FinetuningRunner
from nat.finetuning.backends.openpipe_art.trajectory_builder import (
    ARTTrajectoryBuilder
)
from nat.finetuning.backends.openpipe_art.trainer_adapter import (
    ARTTrainerAdapter
)
from nat.finetuning.backends.openpipe_art.config import ARTTrainerConfig
from nat.data_models.finetuning import (
    CurriculumLearningConfig,
    TrainerRunConfig,
    TrainingJobRef,
    TrainingJobStatus,
    TrainingStatusEnum
)

# Configure matplotlib for non-interactive backend
try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

logger = logging.getLogger(__name__)


class OpenPipeARTRunner(FinetuningRunner):
    """
    Concrete implementation of FinetuningRunner for the OpenPipe ART backend.

    This runner orchestrates the finetuning process using:
    - ARTTrajectoryBuilder to collect trajectories from evaluations
    - ARTTrainerAdapter to submit trajectories to the ART training backend
    """

    def __init__(
            self,
            trainer_config: ARTTrainerConfig,
            run_config: TrainerRunConfig,
            num_generations: int = 1,
            output_dir: Path | None = None,
            curriculum_config: CurriculumLearningConfig | None = None
    ):
        """
        Initialize the OpenPipe ART Runner.

        Args:
            trainer_config: Configuration for the ART trainer backend
            run_config: Configuration for the training run
            num_generations: Number of trajectory generations per example
            output_dir: Output directory for logs and plots
            curriculum_config: Optional curriculum learning configuration
        """
        super().__init__(
            trainer_config,
            run_config,
            backend="openpipe_art",
            curriculum_config=curriculum_config
        )

        # Type hint for the specific config
        self.trainer_config: ARTTrainerConfig = trainer_config
        self.num_generations = num_generations

        # Initialize components
        self.trajectory_builder = ARTTrajectoryBuilder(trainer_config, run_config, num_generations)
        self.trainer_adapter = ARTTrainerAdapter(trainer_config, run_config, verbose=True)

        # Track job references
        self._job_refs: list[TrainingJobRef] = []
        self._run_id: str | None = None

        # Track rewards for plotting
        self._reward_history: list[dict] = []
        self._validation_history: list[dict] = []
        self._output_dir = output_dir or Path("./.tmp/nat/finetuning/outputs")

    async def initialize(self) -> None:
        """
        Initialize the runner and its components.

        This will:
        - Initialize the TrainerAdapter and verify connectivity
        - Prepare the TrajectoryBuilder for collecting trajectories
        """
        logger.info("Initializing OpenPipe ART Runner")

        # Initialize the trainer adapter
        await self.trainer_adapter.initialize()

        # Generate a unique run ID
        self._run_id = f"art_run_{uuid.uuid4().hex[:8]}"

        logger.info(f"OpenPipe ART Runner initialized with run ID: {self._run_id}")

    async def run_epoch(self, epoch: int, run_id: str) -> TrainingJobRef | None:
        """
        Run a single epoch of training.

        Args:
            epoch: The current epoch number (0-indexed)
            run_id: Unique identifier for this training run

        Returns:
            TrainingJobRef: Reference to the submitted training job
        """
        logger.info(f"Starting epoch {epoch} for run {run_id}")

        # Start the trajectory builder for this epoch
        epoch_meta = {
            "epoch": epoch,
            "run_id": run_id,
            "trainer_config": self.trainer_config.model_dump(),
        }

        # Check if we should run validation
        if (
                self.run_config.validation_dataset
                and epoch % self.run_config.validation_interval == 0
        ):
            logger.info(f"Running validation at epoch {epoch + 1}")
            validation_metrics = await self.run_validation_evaluation(
                epoch,
                self._run_id,
                self.run_config.validation_dataset
            )

            # Store validation metrics
            validation_info = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "avg_reward": validation_metrics.get("avg_reward", 0.0),
                "min_reward": validation_metrics.get("min_reward", 0.0),
                "max_reward": validation_metrics.get("max_reward", 0.0),
                "num_examples": validation_metrics.get("num_examples", 0),
            }
            self._validation_history.append(validation_info)


        await self.trajectory_builder.start_run(run_id=run_id, meta=epoch_meta)

        # Finalize and get trajectories
        trajectory_collection = await self.trajectory_builder.finalize(run_id=run_id, meta=epoch_meta)

        if not trajectory_collection.trajectories:
            logger.warning(f"No trajectories collected for epoch {epoch}")
            # Return a dummy job ref
            return None

        # Calculate metrics from the original trajectories (before curriculum filtering)
        # trajectory_collection.trajectories is a list of lists
        # Each inner list contains trajectories for a specific example
        all_rewards = []
        total_trajectories = 0
        group_stats = []

        for trajectory_list in trajectory_collection.trajectories:
            group_rewards = []
            for trajectory in trajectory_list:
                total_trajectories += 1
                if hasattr(trajectory, 'reward'):
                    reward = trajectory.reward
                    all_rewards.append(reward)
                    group_rewards.append(reward)

            if group_rewards:
                avg_group_reward = sum(group_rewards) / len(group_rewards)
                variance = sum((r - avg_group_reward) ** 2 for r in group_rewards) / len(group_rewards)
                group_stats.append({
                    "avg_reward": avg_group_reward,
                    "variance": variance,
                    "size": len(group_rewards)
                })

        logger.info(f"Collected {total_trajectories} trajectories in {len(trajectory_collection.trajectories)} groups for epoch {epoch}")

        # Calculate reward statistics from all trajectories
        if all_rewards:
            avg_reward = sum(all_rewards) / len(all_rewards)
            min_reward = min(all_rewards)
            max_reward = max(all_rewards)
        else:
            avg_reward = min_reward = max_reward = 0.0

        # Apply curriculum learning to filter trajectories
        filtered_collection = self.apply_curriculum_learning(trajectory_collection, epoch)

        # Calculate metrics after curriculum filtering
        filtered_trajectories = 0
        filtered_rewards = []
        for trajectory_list in filtered_collection.trajectories:
            for trajectory in trajectory_list:
                filtered_trajectories += 1
                if hasattr(trajectory, 'reward'):
                    filtered_rewards.append(trajectory.reward)

        if filtered_rewards:
            filtered_avg_reward = sum(filtered_rewards) / len(filtered_rewards)
            filtered_min_reward = min(filtered_rewards)
            filtered_max_reward = max(filtered_rewards)
        else:
            filtered_avg_reward = filtered_min_reward = filtered_max_reward = 0.0

        # Log progress with both original and filtered metrics
        metrics = {
            "avg_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "num_trajectories": total_trajectories,
            "num_groups": len(trajectory_collection.trajectories),
            # Curriculum metrics
            "filtered_trajectories": filtered_trajectories,
            "filtered_groups": len(filtered_collection.trajectories),
            "filtered_avg_reward": filtered_avg_reward,
            "filtered_min_reward": filtered_min_reward,
            "filtered_max_reward": filtered_max_reward,
            "curriculum_percentile": self._curriculum_state["current_percentile"] if self.curriculum_config.enabled else 1.0,
        }

        # Log group statistics if curriculum learning is enabled
        if self.curriculum_config.enabled and group_stats:
            sorted_groups = sorted(group_stats, key=lambda x: x["avg_reward"], reverse=True)
            logger.info(
                "Group reward distribution - Top: %.4f, Median: %.4f, Bottom: %.4f",
                sorted_groups[0]["avg_reward"],
                sorted_groups[len(sorted_groups)//2]["avg_reward"],
                sorted_groups[-1]["avg_reward"]
            )

        self.log_progress(epoch, metrics)

        # Check if we have trajectories after filtering
        if not filtered_collection.trajectories:
            logger.warning(f"No trajectories remaining after curriculum filtering for epoch {epoch}")
            return None

        # Submit filtered trajectories to trainer
        job_ref = await self.trainer_adapter.submit(filtered_collection)
        self._job_refs.append(job_ref)

        logger.info(f"Submitted training job for epoch {epoch}: {job_ref}")

        return job_ref

    async def run(self, num_epochs: int) -> list[TrainingJobStatus]:
        """
        Run the complete finetuning workflow for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            list[TrainingJobStatus]: Status of all training jobs
        """
        if not self._run_id:
            await self.initialize()

        logger.info(f"Starting finetuning run with {num_epochs} epochs")

        job_statuses = []

        for epoch in range(num_epochs):
            try:
                # Run the epoch
                job_ref = await self.run_epoch(epoch, self._run_id)

                # Wait for completion before starting next epoch
                if job_ref:
                    status = await self.trainer_adapter.wait_until_complete(job_ref)
                    job_statuses.append(status)

                    # Check if training failed
                    if status.status == TrainingStatusEnum.FAILED:
                        logger.error(f"Training failed at epoch {epoch}: {status.message}")
                        break
                else:
                    # No trajectories collected, create a dummy status
                    job_statuses.append(TrainingJobStatus(
                        run_id=self._run_id,
                        backend=self.backend,
                        status=TrainingStatusEnum.COMPLETED,
                        message="No trajectories to train on",
                        metadata={"epoch": epoch}
                    ))

                logger.info(f"Completed epoch {epoch + 1}/{num_epochs}")

            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {e}")
                job_statuses.append(TrainingJobStatus(
                    run_id=self._run_id,
                    backend=self.backend,
                    status=TrainingStatusEnum.FAILED,
                    message=str(e),
                    metadata={"epoch": epoch}
                ))
                break

        logger.info(f"Finetuning run completed. Processed {len(job_statuses)} epochs")
        return job_statuses

    async def get_metrics(self, run_id: str) -> dict[str, Any]:
        """
        Get training metrics for a specific run.

        Args:
            run_id: The run identifier

        Returns:
            dict: Metrics from the training run
        """
        metrics = {
            "run_id": run_id,
            "total_epochs": len(self._job_refs),
            "jobs": []
        }

        for job_ref in self._job_refs:
            try:
                status = await self.trainer_adapter.status(job_ref)
                metrics["jobs"].append({
                    "job_ref": job_ref.model_dump(),
                    "status": status.model_dump()
                })
            except Exception as e:
                logger.error(f"Failed to get status for job {job_ref}: {e}")
                metrics["jobs"].append({
                    "job_ref": job_ref.model_dump(),
                    "error": str(e)
                })

        return metrics

    async def cleanup(self) -> None:
        """
        Clean up any resources used by the runner.
        """
        logger.info("Cleaning up OpenPipe ART Runner resources")

        # Cleanup trajectory builder tasks
        if hasattr(self.trajectory_builder, 'evaluation_runs'):
            for run_id, task in self.trajectory_builder.evaluation_runs.items():
                if not task.done():
                    logger.info(f"Cancelling evaluation task for run {run_id}")
                    task.cancel()

        # Cleanup trainer adapter tasks
        if hasattr(self.trainer_adapter, 'training_jobs'):
            for job_id, task in self.trainer_adapter.training_jobs.items():
                if not task.done():
                    logger.info(f"Cancelling training task for job {job_id}")
                    task.cancel()

        logger.info("OpenPipe ART Runner cleanup completed")

    def log_progress(
            self,
            epoch: int,
            metrics: dict[str, Any],
            output_dir: str | None = None
    ) -> None:
        """
        Log training progress and create visualizations.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
            output_dir: Optional output directory override
        """
        # Use provided output_dir or default
        out_dir = Path(output_dir) if output_dir else self._output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract and store reward info
        reward_info = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "avg_reward": metrics.get("avg_reward", 0.0),
            "min_reward": metrics.get("min_reward", 0.0),
            "max_reward": metrics.get("max_reward", 0.0),
            "num_trajectories": metrics.get("num_trajectories", 0),
        }
        self._reward_history.append(reward_info)

        # Create plots
        self._create_reward_plot(epoch, out_dir)

        # Log metrics to JSON file
        self._log_metrics_to_file(epoch, metrics, out_dir)

        logger.info(
            "Epoch %d progress logged - Avg Reward: %.4f, Trajectories: %d",
            epoch,
            reward_info["avg_reward"],
            reward_info["num_trajectories"]
        )

    def _create_reward_plot(self, epoch: int, output_dir: Path) -> None:
        """Create PNG plot showing reward progression and curriculum learning status."""
        if not self._reward_history:
            return

        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not available, skipping plot generation"
            )
            return

        # Create figure with potentially two y-axes
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot training rewards
        epochs = [r["epoch"] for r in self._reward_history]
        avg_rewards = [r["avg_reward"] for r in self._reward_history]

        ax.plot(epochs, avg_rewards, 'b-', linewidth=2, label='Training Average Reward')
        ax.scatter(epochs, avg_rewards, s=50, c='blue', zorder=5)

        # Plot filtered average rewards if curriculum learning is enabled
        if self.curriculum_config.enabled and any("filtered_avg_reward" in r for r in self._reward_history):
            filtered_avg_rewards = [r.get("filtered_avg_reward", r["avg_reward"]) for r in self._reward_history]
            ax.plot(epochs, filtered_avg_rewards, 'g:', linewidth=2, label='Filtered Avg Reward (Curriculum)')
            ax.scatter(epochs, filtered_avg_rewards, s=30, c='green', zorder=4)

        # Plot validation rewards if available
        val_epochs = []
        val_avg_rewards = []
        if self._validation_history:
            val_epochs = [r["epoch"] for r in self._validation_history]
            val_avg_rewards = [r["avg_reward"] for r in self._validation_history]

            ax.plot(val_epochs, val_avg_rewards, 'r--', linewidth=2, label='Validation Average Reward')
            ax.scatter(val_epochs, val_avg_rewards, s=50, c='red', zorder=5)

            # Combine all rewards for y-axis range calculation
            all_rewards = avg_rewards + val_avg_rewards
        else:
            all_rewards = avg_rewards

        # Calculate y-axis range with margin
        if all_rewards:
            min_avg = min(all_rewards)
            max_avg = max(all_rewards)
            # Add 10% margin on each side
            range_margin = (max_avg - min_avg) * 0.1
            # If all rewards are the same, use a fixed margin
            if range_margin == 0:
                range_margin = abs(min_avg) * 0.1 if min_avg != 0 else 0.1
            ax.set_ylim(min_avg - range_margin, max_avg + range_margin)

        # Add curriculum learning progression on secondary y-axis if enabled
        if self.curriculum_config.enabled:
            ax2 = ax.twinx()
            curriculum_percentiles = [r.get("curriculum_percentile", 1.0) * 100 for r in self._reward_history]
            ax2.plot(epochs, curriculum_percentiles, 'm-.', linewidth=1.5, label='Curriculum %', alpha=0.7)
            ax2.set_ylabel('Curriculum Percentile (%)', fontsize=11, color='m')
            ax2.set_ylim(0, 105)
            ax2.tick_params(axis='y', labelcolor='m')
            ax2.grid(False)

            # Add shaded regions to indicate curriculum expansions
            expansion_epochs = []
            for i in range(1, len(curriculum_percentiles)):
                if curriculum_percentiles[i] > curriculum_percentiles[i-1]:
                    expansion_epochs.append(epochs[i])

            for exp_epoch in expansion_epochs:
                ax.axvline(x=exp_epoch, color='purple', linestyle=':', alpha=0.3, linewidth=1)

        # Formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)

        title = f'Training Progress - Epoch {epoch}'
        if self.curriculum_config.enabled:
            title += f' (Curriculum Learning: {self._curriculum_state["current_percentile"]*100:.1f}%)'
        ax.set_title(title, fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Set integer x-axis ticks
        ax.set_xticks(epochs)

        # Add value annotations for training (reduced to avoid clutter)
        # Only annotate every 5th epoch if there are more than 10 epochs
        annotation_epochs = epochs if len(epochs) <= 10 else epochs[::5]

        for e in annotation_epochs:
            idx = epochs.index(e)
            ax.annotate(f'{avg_rewards[idx]:.3f}',
                        (e, avg_rewards[idx]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        color='blue')

        # Add value annotations for validation (sparse)
        if self._validation_history:
            val_annotation_epochs = val_epochs if len(val_epochs) <= 5 else val_epochs[::2]
            for e in val_annotation_epochs:
                idx = val_epochs.index(e)
                ax.annotate(f'{val_avg_rewards[idx]:.3f}',
                            (e, val_avg_rewards[idx]),
                            textcoords="offset points",
                            xytext=(0, -15),
                            ha='center',
                            fontsize=8,
                            color='red')

        # Save plot
        plot_path = output_dir / "reward_plot.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.debug("Saved reward plot to %s", plot_path)

    def _create_trajectory_builder(self, run_config: TrainerRunConfig):
        """
        Create a trajectory builder instance for the ART backend.

        Args:
            run_config: Configuration for the run

        Returns:
            ARTTrajectoryBuilder: Instance of trajectory builder
        """
        return ARTTrajectoryBuilder(
            self.trainer_config,
            run_config,
            self.num_generations
        )

    def _log_metrics_to_file(self, epoch: int, metrics: dict[str, Any], output_dir: Path) -> None:
        """Log metrics to JSON file."""
        # Create metrics log file
        metrics_file = output_dir / "training_metrics.jsonl"

        # Prepare log entry
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "run_id": self._run_id,
            **metrics
        }

        # Add curriculum learning state if enabled
        if self.curriculum_config.enabled:
            log_entry["curriculum_state"] = self.get_curriculum_state()

        # Append to file
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Also save reward history separately
        history_file = output_dir / "reward_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self._reward_history, f, indent=2)

        # Save validation history if available
        if self._validation_history:
            val_history_file = output_dir / "validation_history.json"
            with open(val_history_file, 'w', encoding='utf-8') as f:
                json.dump(self._validation_history, f, indent=2)

        # Save curriculum learning history if enabled
        if self.curriculum_config.enabled:
            curriculum_file = output_dir / "curriculum_state.json"
            with open(curriculum_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_curriculum_state(), f, indent=2)
