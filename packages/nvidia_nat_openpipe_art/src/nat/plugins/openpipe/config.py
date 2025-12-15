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

import art
from pydantic import BaseModel
from pydantic import Field

from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig


class ARTTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="openpipe_art_traj_builder"):
    """
    Configuration for the OpenPipe ART Trajectory Builder.
    """
    num_generations: int = Field(default=2,
                                 description="Number of trajectory generations per example in eval dataset",
                                 ge=1)


class ARTBackendConfig(BaseModel):
    """
    Base configuration for the ART backend.
    """
    ip: str = Field(description="IP Address of Remote Backend")

    port: int = Field(description="Port for Remote Backend")

    name: str = Field(default="trainer_run", description="Name of the Trainer run.")

    project: str = Field(default="trainer_project", description="Project name for the Trainer run.")

    base_model: str = Field(
        description="Base model to use for the training. This is the model that will be fine-tuned.",
        default="Qwen/Qwen2.5-7B-Instruct")

    api_key: str = Field(description="API key for authenticating with the ART backend.", default="default")

    delete_old_checkpoints: bool = Field(description="Whether to delete old checkpoints after a training epoch",
                                         default=False)

    init_args: art.dev.InitArgs | None = Field(description="Initialization args for Remote Backend", default=None)

    engine_args: art.dev.EngineArgs | None = Field(description="Engine args for Remote Backend", default=None)

    torchtune_args: art.dev.TorchtuneArgs | None = Field(description="Torchtune args for Remote Backend", default=None)

    server_config: art.dev.OpenAIServerConfig | None = Field(description="Server args for Remote Backend", default=None)


class ARTTrainerAdapterConfig(TrainerAdapterConfig, name="openpipe_art_trainer_adapter"):
    """
    Configuration for the ART Trainer run
    """

    backend: ARTBackendConfig = Field(description="Configuration for the ART backend.")
    training: art.dev.TrainerArgs | None = Field(description="Training args for Remote Backend", default=None)


class ARTTrainerConfig(TrainerConfig, name="openpipe_art_trainer"):
    """
    Configuration for the ART Trainer run
    """
    pass
