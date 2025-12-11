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

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import art
import httpx

from nat.data_models.finetuning import EpisodeItem
from nat.data_models.finetuning import EpisodeItemRole
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter

from .config import ARTTrainerAdapterConfig

logger = logging.getLogger(__name__)


class ARTTrainerAdapter(TrainerAdapter):
    """
    Adapter for the ART Trainer backend.
    """

    def __init__(self, adapter_config: ARTTrainerAdapterConfig):
        super().__init__(adapter_config)

        self.adapter_config: ARTTrainerAdapterConfig = adapter_config

        self.remote_backend: art.Backend = art.Backend(
            base_url=f"http://{adapter_config.backend.ip}:{adapter_config.backend.port}")

        self._model_internal_config: art.dev.InternalModelConfig = art.dev.InternalModelConfig(
            init_args=self.adapter_config.backend.init_args,
            engine_args=self.adapter_config.backend.engine_args,
            torchtune_args=self.adapter_config.backend.torchtune_args,
            trainer_args=self.adapter_config.training)

        self.model: art.TrainableModel = art.TrainableModel(
            name=self.adapter_config.backend.name,
            project=self.adapter_config.backend.project,
            base_model=self.adapter_config.backend.base_model,
            _internal_config=self._model_internal_config,
        )

        self._training_jobs: dict[str, asyncio.Task[None]] = {}

        logger.info(f"Initialized ARTTrainerAdapter with model: {self.model}")

    @property
    def training_jobs(self) -> dict[str, asyncio.Task[None]]:
        return self._training_jobs

    async def initialize(self, run_config: FinetuneConfig) -> None:

        await super().initialize(run_config)

        await self.model.register(self.remote_backend, _openai_client_config=self.adapter_config.backend.server_config)

        health = await self.is_healthy()

        if not health:
            raise ConnectionError("Failed to connect to ART backend.")

        logger.info("Successfully registered with ART backend.")

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient() as c:
                await c.get(f"http://{self.adapter_config.backend.ip}:8000/v1/models",
                            headers={"Authorization": f"Bearer {self.adapter_config.backend.api_key}"})
            return True
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _validate_episode_order(self, traj: Trajectory):
        """
        Checks all EpisodeItem in traj.episode to validate:

        - Every EpisodeItem.role is EpisodeItemRole.USER, SYSTEM, or ASSISTANT
        - The first EpisodeItem.role is SYSTEM or USER
        - The last EpisodeItem.role is ASSISTANT
        - No two consecutive EpisodeItem.role are the same, except for SYSTEM

        Args:
            traj: Trajectory to validate

        Raises:
            ValueError: If any of the above conditions are not met.
        """
        if not traj.episode:
            raise ValueError("Trajectory episode is empty.")

        if traj.episode[0].role not in {EpisodeItemRole.USER, EpisodeItemRole.SYSTEM}:
            raise ValueError("The first message in the trajectory must be from 'user' or 'system'.")

        # if traj.episode[-1].role != EpisodeItemRole.ASSISTANT:
        #     raise ValueError("The last message in the trajectory must be from 'assistant'.")

        for i in range(1, len(traj.episode)):
            if traj.episode[i].role == traj.episode[i - 1].role and traj.episode[i].role == EpisodeItemRole.ASSISTANT:
                raise ValueError("Consecutive assistant messages from the same role found in trajectory.")

    async def _construct_trajectory_groups(self, trajectory_lists: list[list[Trajectory]]) -> list[art.TrajectoryGroup]:
        """
        Convert list of lists of NAT Trajectory to list of ART TrajectoryGroup.

        Args:
            trajectory_lists: List of lists of NAT Trajectory (each inner list
                contains trajectories for one example).

        Returns:
            List of ART TrajectoryGroup.

        Raises:
            ValueError: If any trajectory is invalid.
        """

        from openai.types.chat.chat_completion import Choice

        # ---------- helpers ----------
        def _as_text(obj: Any) -> str:
            return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)

        def _to_chat_msg(d: EpisodeItem) -> dict:

            if d.role == EpisodeItemRole.USER:
                return {
                    "role": "user",
                    "content": _as_text(d.content),
                }
            elif d.role == EpisodeItemRole.SYSTEM:
                return {
                    "role": "system",
                    "content": _as_text(d.content),
                }
            else:
                return {"role": "assistant", "content": _as_text(d.content)}

        output_trajectory_groups = []

        for trajectory_list in trajectory_lists:
            art_trajectories = []

            for traj in trajectory_list:
                episode = traj.episode
                reward = traj.reward

                # Validate episode order
                await self._validate_episode_order(traj)

                try:
                    first_msg = _to_chat_msg(episode[0])

                    t = art.Trajectory(messages_and_choices=[first_msg], reward=reward)

                    for msg in episode[1:]:
                        if msg.role == EpisodeItemRole.ASSISTANT:
                            t.messages_and_choices.append(
                                Choice(index=0, logprobs=msg.logprobs, message=_to_chat_msg(msg), finish_reason="stop"))
                        else:
                            t.messages_and_choices.append(_to_chat_msg(msg))

                    # Sanity check: art.Trajectory.model_validate()
                    t.model_validate(t.model_dump())

                    art_trajectories.append(t)

                except Exception as e:
                    logger.error(f"Error constructing trajectory: {e}. Skipping.")
                    continue

            # Create TrajectoryGroup for this list of trajectories
            if art_trajectories:
                trajectory_group = art.TrajectoryGroup(trajectories=art_trajectories)
                output_trajectory_groups.append(trajectory_group)

        return output_trajectory_groups

    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """
        Submit trajectories to ART backend for training.

        Args:
            trajectories: TrajectoryCollection with list of lists of NAT Trajectory.

        Returns:
            TrainingJobRef: Reference to the submitted training job.
        """

        trajectory_groups = await self._construct_trajectory_groups(trajectories.trajectories)
        if not trajectory_groups:
            raise ValueError("No valid trajectory groups to submit.")

        assert trajectories.run_id not in self.training_jobs, (f"Training job "
                                                               f"with run_id {trajectories.run_id} already exists.")

        # Delete old remote checkpoints
        if self.adapter_config.backend.delete_old_checkpoints:
            try:
                logger.info("Deleting old checkpoints on ART backend...")
                await self.model.delete_checkpoints()
            except Exception as e:
                logger.warning(f"Failed to delete old checkpoints: {e}")

        # Submit new trajectories
        task = asyncio.create_task(
            self.model.train(trajectory_groups=trajectory_groups,
                             verbose=False,
                             config=art.types.TrainConfig(
                                 beta=getattr(self.adapter_config.training, "beta", 0),
                                 learning_rate=getattr(self.adapter_config.training, "learning_rate", 5e-5),
                             )),
            name=f"art-train:{trajectories.run_id}",
        )

        # Optional: log + cleanup on completion to avoid leaks
        def _on_done(t: asyncio.Task, rid: str = trajectories.run_id) -> None:
            if t.cancelled():
                logger.info(f"Training {rid} was cancelled.")
            elif (exc := t.exception()) is not None:
                logger.exception(f"Training {rid} failed", exc_info=exc)
            else:
                logger.info(f"Training {rid} completed successfully.")

        task.add_done_callback(_on_done)

        self.training_jobs[trajectories.run_id] = task

        total_trajectories = sum(len(group.trajectories) for group in trajectory_groups)
        logger.info(f"Submitted {total_trajectories} trajectories in {len(trajectory_groups)} groups for "
                    f"training with run_id {trajectories.run_id}.")

        return TrainingJobRef(run_id=trajectories.run_id, backend="openpipe-art")

    async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
        task = self.training_jobs.get(ref.run_id)
        if task is None:
            raise ValueError(f"No training job found with run_id {ref.run_id}.")

        if task.done():
            if task.cancelled():
                status = TrainingStatusEnum.CANCELED
                progress = None
                message = "Training was cancelled."
            else:
                exc = task.exception()
                if exc is not None:
                    status = TrainingStatusEnum.FAILED
                    progress = None
                    message = f"Training failed with error: {exc!r}"
                else:
                    status = TrainingStatusEnum.COMPLETED
                    progress = 100.0
                    message = "Training completed successfully."

            _ = self.training_jobs.pop(ref.run_id, None)  # Clean up completed job

        else:
            status = TrainingStatusEnum.RUNNING
            progress = None
            message = "Training is in progress."

        return TrainingJobStatus(
            run_id=ref.run_id,
            backend=ref.backend,
            status=status,
            progress=progress,
            message=message,
        )

    async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float = 10.0) -> TrainingJobStatus:
        task = self.training_jobs.get(ref.run_id)
        if task is None:
            raise ValueError(f"No training job found with run_id {ref.run_id}.")

        while not task.done():
            await asyncio.sleep(poll_interval)

        return await self.status(ref)

    def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """
        Log training adapter progress.

        Args:
            ref: Training job reference
            metrics: Dictionary of metrics to log
            output_dir: Optional output directory override
        """
        # Use default output directory if not provided
        out_dir = Path(output_dir) if output_dir else Path("./.tmp/nat/finetuning/trainer_adapter")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create log file for trainer adapter
        log_file = out_dir / f"trainer_adapter_{ref.run_id}.jsonl"

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": ref.run_id,
            "backend": ref.backend,
            "trainer_config": {
                "base_model": self.adapter_config.backend.base_model,
                "project": self.adapter_config.backend.project,
                "name": self.adapter_config.backend.name,
            },
            **metrics
        }

        # Append to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info("Trainer adapter progress logged for job %s: status=%s",
                    ref.run_id,
                    metrics.get("status", "unknown"))
