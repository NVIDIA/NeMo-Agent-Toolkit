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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_trainer
from nat.cli.register_workflow import register_trainer_adapter
from nat.cli.register_workflow import register_trajectory_builder

from .config import ARTTrainerAdapterConfig
from .config import ARTTrainerConfig
from .config import ARTTrajectoryBuilderConfig
from .trainer import ARTTrainer
from .trainer_adapter import ARTTrainerAdapter
from .trajectory_builder import ARTTrajectoryBuilder


@register_trajectory_builder(config_type=ARTTrajectoryBuilderConfig)
async def register_art_trajectory_builder(config: ARTTrajectoryBuilderConfig, builder: Builder):
    """
    Register the ART trajectory builder.

    Args:
        config: TrajectoryBuilderConfig object
        builder: Builder instance

    Returns:
        ARTTrajectoryBuilder instance
    """
    yield ARTTrajectoryBuilder(trajectory_builder_config=config)


@register_trainer_adapter(config_type=ARTTrainerAdapterConfig)
async def register_art_trainer_adapter(config: ARTTrainerAdapterConfig, builder: Builder):
    """
    Register the ART trainer adapter.

    Args:
        config: TrainerAdapterConfig object
        builder: Builder instance

    Returns:
        ARTTrainerAdapter instance
    """
    yield ARTTrainerAdapter(adapter_config=config)


@register_trainer(config_type=ARTTrainerConfig)
async def register_art_trainer(config: ARTTrainerConfig, builder: Builder):
    """
    Register the ART trainer.

    Args:
        config: TrainerConfig object
        builder: Builder instance

    Returns:
        ARTTrainer instance
    """
    yield ARTTrainer(trainer_config=config)
