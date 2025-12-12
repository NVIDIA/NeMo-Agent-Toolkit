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
Registration module for DPO components.

This module registers the DPO trajectory builder and NeMo Customizer trainer adapter
with NAT's finetuning harness:
- `_type: dpo_traj_builder` - DPO Trajectory Builder
- `_type: nemo_customizer_trainer_adapter` - NeMo Customizer TrainerAdapter
"""

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_trainer
from nat.cli.register_workflow import register_trainer_adapter
from nat.cli.register_workflow import register_trajectory_builder

from .config import DPOTrajectoryBuilderConfig
from .config import NeMoCustomizerTrainerAdapterConfig
from .config import NeMoCustomizerTrainerConfig
from .trainer import NeMoCustomizerTrainer
from .trainer_adapter import NeMoCustomizerTrainerAdapter
from .trajectory_builder import DPOTrajectoryBuilder


@register_trajectory_builder(config_type=DPOTrajectoryBuilderConfig)
async def dpo_trajectory_builder(config: DPOTrajectoryBuilderConfig, builder: Builder):
    """
    Register the DPO (Direct Preference Optimization) trajectory builder.

    This builder collects preference data from workflows that produce scored
    candidate intermediate steps (TTC_END events with TTCEventData).

    The builder:
    1. Runs evaluation to collect intermediate steps
    2. Filters for TTC_END steps with the configured name
    3. Groups candidates by turn_id
    4. Generates preference pairs based on score differences
    5. Builds trajectories with DPOItem episodes

    Example YAML configuration::

        trajectory_builders:
          dpo_builder:
            _type: dpo_traj_builder
            ttc_step_name: dpo_candidate_move
            exhaustive_pairs: true
            min_score_diff: 0.05
            max_pairs_per_turn: 5

        finetuning:
          enabled: true
          trajectory_builder: dpo_builder
          # ... other finetuning config

    Args:
        config: The trajectory builder configuration.
        builder: The NAT workflow builder (for accessing other components).

    Yields:
        A configured DPOTrajectoryBuilder instance.
    """
    yield DPOTrajectoryBuilder(trajectory_builder_config=config)


@register_trainer_adapter(config_type=NeMoCustomizerTrainerAdapterConfig)
async def nemo_customizer_trainer_adapter(config: NeMoCustomizerTrainerAdapterConfig, builder: Builder):
    """
    Register the NeMo Customizer trainer adapter.

    This adapter submits DPO/SFT training jobs to NeMo Customizer and
    optionally deploys the trained model.

    The adapter:
    1. Converts trajectories to JSONL format for DPO training
    2. Uploads datasets to NeMo Datastore
    3. Submits customization jobs to NeMo Customizer
    4. Monitors job progress and status
    5. Optionally deploys trained models

    Example YAML configuration::

        trainer_adapters:
          nemo_customizer:
            _type: nemo_customizer_trainer_adapter
            entity_host: https://nmp.example.com
            datastore_host: https://datastore.example.com
            namespace: my-project
            customization_config: meta/llama-3.2-1b-instruct@v1.0.0+A100
            hyperparameters:
              training_type: dpo
              epochs: 5
              batch_size: 8
            use_full_message_history: true
            deploy_on_completion: true

        finetuning:
          enabled: true
          trainer_adapter: nemo_customizer
          # ... other finetuning config

    Args:
        config: The trainer adapter configuration.
        builder: The NAT workflow builder (for accessing other components).

    Yields:
        A configured NeMoCustomizerTrainerAdapter instance.
    """
    yield NeMoCustomizerTrainerAdapter(adapter_config=config)


@register_trainer(config_type=NeMoCustomizerTrainerConfig)
async def nemo_customizer_trainer(config: NeMoCustomizerTrainerConfig, builder: Builder):
    """
    Register the NeMo Customizer trainer.

    This trainer orchestrates DPO data collection and training job submission.
    Unlike epoch-based trainers, it:
    1. Runs the trajectory builder multiple times (num_runs) to collect data
    2. Aggregates all trajectories into a single dataset
    3. Submits the dataset to NeMo Customizer for training
    4. Monitors the training job until completion

    Example YAML configuration::

        trainers:
          nemo_dpo:
            _type: nemo_customizer_trainer
            num_runs: 5
            wait_for_completion: true
            deduplicate_pairs: true
            max_pairs: 10000

        finetuning:
          enabled: true
          trainer: nemo_dpo
          # ... other finetuning config

    Args:
        config: The trainer configuration.
        builder: The NAT workflow builder (for accessing other components).

    Yields:
        A configured NeMoCustomizerTrainer instance.
    """
    yield NeMoCustomizerTrainer(trainer_config=config)
