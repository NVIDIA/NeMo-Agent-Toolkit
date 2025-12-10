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
Configuration classes for the DPO Trajectory Builder.

This module provides configuration for collecting preference data from workflows
that produce scored TTC (Test-Time Compute) intermediate steps with TTCEventData.
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.finetuning import TrajectoryBuilderConfig


class DPOTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="dpo_traj_builder"):
    """
    Configuration for the DPO (Direct Preference Optimization) Trajectory Builder.

    This builder collects preference pairs from workflows that produce TTC_END
    intermediate steps with TTCEventData. It uses the structured TTCEventData
    model to extract turn_id, candidate_index, score, input (prompt), and
    output (response) - no dictionary key configuration needed.

    The builder groups candidates by turn_id and creates preference pairs based
    on score differences.

    Example YAML configuration:
    ```yaml
    trajectory_builders:
      dpo_builder:
        _type: dpo_traj_builder
        ttc_step_name: dpo_candidate_move
        exhaustive_pairs: true
        min_score_diff: 0.05
        max_pairs_per_turn: 5
    ```
    """

    # === Step Filtering ===
    ttc_step_name: str = Field(
        default="dpo_candidate_move",
        description="Name of the TTC intermediate step to collect. "
        "The builder filters for TTC_END events with this name.",
    )

    # === Pair Generation Modes ===
    exhaustive_pairs: bool = Field(
        default=True,
        description="If True, generate all pairwise comparisons where "
        "score(A) > score(B). If False, only generate best vs worst pair.",
    )

    min_score_diff: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum score difference required to create a preference "
        "pair. Pairs with smaller differences are filtered out.",
    )

    max_pairs_per_turn: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of preference pairs to generate per turn. "
        "If None, no limit. Pairs sorted by score difference (highest first).",
    )

    # === Reward Computation ===
    reward_from_score_diff: bool = Field(
        default=True,
        description="If True, compute trajectory reward as score difference "
        "(chosen - rejected). If False, use chosen score directly as reward.",
    )

    # === Validation ===
    require_multiple_candidates: bool = Field(
        default=True,
        description="If True, skip turns with only one candidate (no preference "
        "signal). If False, include single-candidate turns.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "DPOTrajectoryBuilderConfig":
        """Validate configuration consistency."""
        if self.max_pairs_per_turn is not None and self.max_pairs_per_turn < 1:
            raise ValueError("max_pairs_per_turn must be at least 1 if specified")
        return self
