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
from typing import Literal

from pydantic import Field, BaseModel
from pathlib import Path

import art

from nat.data_models.finetuning import RLTrainerConfig, TrainerRunConfig, BaseFinetuningConfig

class ARTBackendConfig(BaseModel):
    """
    Base configuration for the ART backend.
    """
    ip: str = Field(description="IP Address of Remote Backend")

    port: int = Field(description="Port for Remote Backend")

    name: str = Field(default="trainer_run_4",
                      description="Name of the Trainer run.")

    project: str = Field(default="trainer_project",
                         description="Project name for the Trainer run.")

    base_model: str = Field(
        description="Base model to use for the training. This is the model that will be fine-tuned.",
        default="Qwen/Qwen2.5-7B-Instruct"
    )

    api_key: str = Field(description="API key for authenticating with the ART backend.",
                                default="default")

    init_args: art.dev.InitArgs | None = Field(description="Initialization args for Remote Backend",
                                        default=None)

    engine_args: art.dev.EngineArgs | None= Field(description="Engine args for Remote Backend",
                                            default=None)

    torchtune_args: art.dev.TorchtuneArgs | None = Field(description="Torchtune args for Remote Backend",
                                                default=None)

    server_config: art.dev.OpenAIServerConfig | None = Field(description="Server args for Remote Backend",
                                                default=None)


class ARTTrainerConfig(RLTrainerConfig):
    """
    Configuration for the ART Trainer run
    """

    backend: ARTBackendConfig = Field(description="Configuration for the ART backend.")
    training: art.dev.TrainerArgs | None = Field(description="Training args for Remote Backend",
                                          default=None)


class ARTFinetuningRunnerConfig(BaseModel):
    """
    Configuration for a FinetuningRunner instance.
    """
    num_epochs: int = Field(default=1, description="Number of epochs to run", ge=1)
    num_generations: int = Field(
        default=2,
        description="Number of trajectory generations per example in eval dataset", 
        ge=1
    )
    checkpoint_interval: int | None = Field(default=None, description="Save checkpoints every N epochs")
    save_best_checkpoint: bool = Field(default=True, description="Save the best checkpoint based on validation metrics")
    early_stopping_patience: int | None = Field(default=None,
                                                description="Stop training if no improvement for N epochs")
    metrics_log_interval: int = Field(default=1, description="Log metrics every N epochs")

    # Trainer and run configurations
    trainer_config: ARTTrainerConfig = Field(description="Configuration for the trainer backend")
    run_config: TrainerRunConfig = Field(description="Configuration for each training run")

    # Optional configurations
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    output_dir: Path = Field(default=Path("./.tmp/nat/finetuning/"),
                             description="Directory for outputs and checkpoints")
    verbose: bool = Field(default=True, description="Enable verbose logging")



class OpenPipeARTFinetuningConfig(BaseFinetuningConfig):
    """
    Configuration for the OpenPipe ART finetuning runner.

    This configuration is used in the config.yml under the 'finetuning' section.
    """
    type: Literal["openpipe_art"] = "openpipe_art"
    # Runner configuration
    runner_config: ARTFinetuningRunnerConfig = Field(
        description="Configuration for the finetuning runner"
    )

    # Optional overrides
    output_dir: Path | None = Field(
        default=None,
        description="Override the output directory for this specific runner"
    )

    description: str = Field(
        default="OpenPipe ART finetuning runner for training models with reinforcement learning",
        description="Description of this finetuning configuration"
    )

    tags: list[str] = Field(
        default_factory=lambda: ["openpipe", "art", "reinforcement-learning", "finetuning"],
        description="Tags for categorizing this configuration"
    )


