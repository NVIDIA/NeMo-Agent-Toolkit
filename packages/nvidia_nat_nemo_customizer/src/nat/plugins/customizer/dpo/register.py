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
Registration module for the DPO Trajectory Builder.

This module registers the DPO trajectory builder with NAT's finetuning harness,
making it available as `_type: dpo_traj_builder` in YAML configuration.
"""

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_trajectory_builder

from .config import DPOTrajectoryBuilderConfig
from .dpo_trajectory_builder import DPOTrajectoryBuilder


@register_trajectory_builder(config_type=DPOTrajectoryBuilderConfig)
async def dpo_trajectory_builder(
    config: DPOTrajectoryBuilderConfig, builder: Builder
):
    """
    Register the DPO (Direct Preference Optimization) trajectory builder.

    This builder collects preference data from workflows that produce scored
    candidate intermediate steps (e.g., dpo_candidate_move steps from the
    DPO Tic-Tac-Toe example).

    The builder:
    1. Runs evaluation to collect intermediate steps
    2. Filters for CUSTOM steps with the configured name
    3. Groups candidates by turn_id
    4. Generates preference pairs based on score differences
    5. Builds trajectories with chosen/rejected responses

    Example YAML configuration:
    ```yaml
    trajectory_builders:
      dpo_builder:
        _type: dpo_traj_builder
        custom_step_name: dpo_candidate_move
        exhaustive_pairs: true
        min_score_diff: 0.05
        max_pairs_per_turn: 5

    finetuning:
      enabled: true
      trajectory_builder: dpo_builder
      # ... other finetuning config
    ```

    Args:
        config: The trajectory builder configuration.
        builder: The NAT workflow builder (for accessing other components).

    Yields:
        A configured DPOTrajectoryBuilder instance.
    """
    yield DPOTrajectoryBuilder(trajectory_builder_config=config)
