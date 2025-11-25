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

from pathlib import Path
from enum import Enum
from typing import Any, Literal

from pydantic import Field
from pydantic import BaseModel, model_validator
import typing

from .common import BaseModelRegistryTag
from .common import TypedBaseModel



class RewardFunctionConfig(BaseModel):
    """
    Configuration for the reward function
    """
    name: str = Field(description="Name of the reward function.")

class TrainerConfig(TypedBaseModel, BaseModelRegistryTag):
    """
    Base configuration for the Trainer
    """
    pass

class TrajectoryBuilderConfig(TypedBaseModel, BaseModelRegistryTag):
    """
    Configuration for the trajectory collector
    """
    pass

class TrainerAdapterConfig(TypedBaseModel, BaseModelRegistryTag):
    """
    Configuration for the trainer adapter
    """
    pass


class FinetuningConfig(BaseModel):
    """
    Configuration for fine-tuning
    """
    enabled: bool = Field(description="Whether fine-tuning is enabled.", default=False)
    trainer: str | None = Field(
        description="The trainer to use for fine-tuning.",
        default=None
    )
    trajectory_builder: str | None = Field(
        description="The trajectory builder to use for fine-tuning.",
        default=None
    )
    trainer_adapter: str | None = Field(
        description="The trainer adapter to use for fine-tuning.",
        default=None
    )
    reward_function: RewardFunctionConfig | None = Field(
        description="Configuration for the reward function.",
        default=None
    )


TrainerConfigT = typing.TypeVar("TrainerConfigT", bound=TrainerConfig)
TrajectoryBuilderConfigT = typing.TypeVar("TrajectoryBuilderConfigT", bound=TrajectoryBuilderConfig)
TrainerAdapterConfigT = typing.TypeVar("TrainerAdapterConfigT", bound=TrainerAdapterConfig)

class TrainingJobRef(BaseModel):
    """
    A reference to a training job.
    """
    run_id: str = Field(description="The ID of the run this job belongs to.")
    backend: str = Field(description="The backend used for the training job.")
    metadata: dict | None = Field(description="Any additional metadata for the training job.", default=None)


class TrainingStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TrainingJobStatus(BaseModel):
    """
    The status of a training job.
    """
    run_id: str = Field(description="The ID of the run this job belongs to.")
    backend: str = Field(description="The backend used for the training job.")
    status: TrainingStatusEnum = Field(description="The current status of the training job.")
    progress: float | None = Field(description="The progress of the training job as a percentage (0.0 to 100.0).",
                                   default=None)
    message: str | None = Field(description="Any additional message or information about the training job.",
                                default=None)
    metadata: dict | None = Field(description="Any additional metadata for the training job.",
                                  default=None)


class EpisodeItemRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    OTHER = "other"


class EpisodeItem(BaseModel):
    """
    A single step in an episode.
    """
    role: EpisodeItemRole = Field(description="The role of the agent (e.g., 'user', 'assistant').")
    content: str = Field(description="The content of the message.")
    logprobs: Any | None = Field(description="The log probabilities of the tokens in the message.",
                                 default=None)
    metadata: dict | None = Field(description="Any additional metadata for the step.", default=None)

    # Add model validator after construction that checks that logprobs can't be none of role is assistant
    @model_validator(mode="after")
    def check_logprobs(self) -> "EpisodeItem":
        if self.role == EpisodeItemRole.ASSISTANT and self.logprobs is None:
            raise ValueError("logprobs must be provided for assistant role.")
        return self


class Trajectory(BaseModel):
    """
    A trajectory is a sequence of states, actions, and rewards.
    """
    episode: list[EpisodeItem] = Field(description="A list of steps in the episode.")
    reward: float = Field(description="The total reward for the episode.")
    shaped_rewards: list[float] | None = Field(description="The shaped rewards for each step in the episode.",
                                               default=None)
    metadata: dict | None = Field(description="Any additional metadata for the trajectory.", default=None)



class TrajectoryCollection(BaseModel):
    """
    A collection of trajectories.
    """
    trajectories: list[list[Trajectory]] = Field(
        description="A list of trajectory lists, each inner list contains trajectories for one example."
    )
    run_id: str = Field(description="The ID of the run this collection belongs to.")


class CurriculumLearningConfig(BaseModel):
    """
    Configuration for curriculum learning in fine-tuning.

    Curriculum learning progressively introduces harder training examples
    to improve model learning and convergence.
    """
    enabled: bool = Field(
        default=False,
        description="Whether to enable curriculum learning"
    )
    initial_percentile: float = Field(
        default=0.3,
        description="Initial percentile of trajectory groups to include (0.0-1.0). "
                    "E.g., 0.3 means start with top 30% easiest groups"
    )
    increment_percentile: float = Field(
        default=0.2,
        description="Percentile increment when expanding curriculum. "
                    "E.g., 0.2 means add 20% more groups each expansion"
    )
    expansion_interval: int = Field(
        default=5,
        description="Number of epochs between curriculum expansions",
        ge=1
    )
    min_reward_diff: float = Field(
        default=0.1,
        description="Minimum reward difference within a group to be included. "
                    "Groups with all same rewards provide no learning signal"
    )
    sort_ascending: bool = Field(
        default=False,
        description="If True, sort groups from low to high reward (hard to easy). "
                    "If False, sort from high to low reward (easy to hard)"
    )

    random_subsample: float | None = Field(
        default=None,
        description="If set, randomly subsample this fraction of trajectories from each group."
    )

    @model_validator(mode="after")
    def validate_percentiles(self) -> "CurriculumLearningConfig":
        """Validate that percentile values are in valid range."""
        if not 0.0 < self.initial_percentile <= 1.0:
            raise ValueError("initial_percentile must be between 0 and 1")
        if not 0.0 < self.increment_percentile <= 1.0:
            raise ValueError("increment_percentile must be between 0 and 1")
        return self


class FinetuneRunConfig(BaseModel):
    """
    Parameters used for a Trainer run
    """

    config_file: Path | BaseModel  # allow for instantiated configs to be passed in
    target_functions: list[str] = ["<workflow>"]
    dataset: str | Path | None = None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    endpoint: str | None = None  # only used when running the workflow remotely
    endpoint_timeout: int = 300
    override: tuple[tuple[str, str], ...] = ()
    validation_dataset: str | Path | None = Field(
        default=None,
        description="Validation dataset file path for periodic validation"
    )

    validation_interval: int = Field(
        default=5,
        description="Run validation every N epochs",
        ge=1
    )

    validation_config_file: str | Path | None = Field(
        default=None,
        description="Optional separate config file for validation runs"
    )

    curriculum_learning: CurriculumLearningConfig = Field(
        default=CurriculumLearningConfig(),
        description="Configuration for curriculum learning during fine-tuning"
    )