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

from abc import ABC
from abc import abstractmethod
from typing import Any

from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrajectoryCollection


class TrainerAdapter(ABC):
    """
    Adapter to send Trajectories to remote training cluster for weights updates.
    """

    def __init__(self, adapter_config: TrainerAdapterConfig):
        self.adapter_config = adapter_config
        self.run_config: FinetuneConfig = None

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """
        Asynchronously initialize any resources needed for the trainer adapter.
        """
        self.run_config = run_config
        self.adapter_config.reward = self.run_config.reward_function

    @abstractmethod
    async def is_healthy(self) -> bool:
        """
        Check the health of the remote training backend.

        Returns:
            bool: True if the backend is healthy, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """
        Submit trajectories to remote training backend.

        Args:
            trajectories (list[Trajectory]): The list of trajectories to submit.

        Returns:
            TrainingJobRef: Reference to the submitted training job.
        """
        raise NotImplementedError

    @abstractmethod
    async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
        """
        Get the status of a submitted training job.

        Args:
            ref (TrainingJobRef): Reference to the training job.

        Returns:
            TrainingJobStatus: The current status of the training job.
        """
        raise NotImplementedError

    @abstractmethod
    async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float = 10.0) -> TrainingJobStatus:
        """
        Wait until the training job is complete.

        Args:
            ref (TrainingJobRef): Reference to the training job.
            poll_interval (float): Time in seconds between status checks.

        Returns:
            TrainingJobStatus: The final status of the training job.
        """
        raise NotImplementedError

    @abstractmethod
    def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """
        Log training adapter progress.

        Args:
            ref: Training job reference
            metrics: Dictionary of metrics to log
            output_dir: Optional output directory override
        """
        raise NotImplementedError
