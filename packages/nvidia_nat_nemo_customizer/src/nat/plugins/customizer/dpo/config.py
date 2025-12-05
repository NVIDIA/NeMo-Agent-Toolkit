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
that produce scored candidate intermediate steps (e.g., dpo_candidate_move steps
from the DPO Tic-Tac-Toe example).
"""

from pydantic import Field
from pydantic import model_validator

from nat.data_models.finetuning import TrajectoryBuilderConfig


class DPOTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="dpo_traj_builder"):
    """
    Configuration for the DPO (Direct Preference Optimization) Trajectory Builder.

    This builder collects preference pairs from workflows that produce scored
    candidate intermediate steps, such as the DPO Tic-Tac-Toe example. It groups
    candidates by turn_id and creates preference pairs based on score differences.

    Example YAML configuration:
    ```yaml
    trajectory_builders:
      dpo_builder:
        _type: dpo_traj_builder
        custom_step_name: dpo_candidate_move
        exhaustive_pairs: true
        min_score_diff: 0.05
        max_pairs_per_turn: 5
    ```
    """

    # === Step Filtering ===
    custom_step_name: str = Field(
        default="dpo_candidate_move",
        description="Name of the CUSTOM intermediate step to collect. "
        "The builder filters for steps with this name.",
    )

    # === Pair Generation Modes ===
    exhaustive_pairs: bool = Field(
        default=True,
        description="If True, generate all pairwise comparisons where score(A) > score(B). "
        "If False, only generate best vs worst pair per turn.",
    )

    min_score_diff: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum score difference required to create a preference pair. "
        "Pairs with smaller differences are filtered out to avoid trivial comparisons.",
    )

    max_pairs_per_turn: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of preference pairs to generate per turn. "
        "If None, no limit is applied. Pairs are sorted by score difference (highest first).",
    )

    # === Prompt Formatting ===
    include_system_prompt: bool = Field(
        default=False,
        description="Whether to prepend the system prompt to the episode. "
        "If True, looks for system_prompt in intermediate step metadata.",
    )

    system_prompt_key: str = Field(
        default="system_prompt",
        description="Metadata key for the system prompt in intermediate steps.",
    )

    # === Field Mappings (for flexibility with different workflows) ===
    turn_id_key: str = Field(
        default="turn_id",
        description="Metadata key for the turn identifier. "
        "Candidates with the same turn_id compete for the same prompt.",
    )

    score_key: str = Field(
        default="score",
        description="Metadata key for the candidate score. "
        "Higher scores indicate better candidates.",
    )

    prompt_key: str = Field(
        default="prompt",
        description="Metadata key for the input prompt. "
        "This is the input that produced the candidate response.",
    )

    response_key: str = Field(
        default="raw_llm_response",
        description="Metadata key for the model response/completion.",
    )

    candidate_index_key: str = Field(
        default="candidate_index",
        description="Metadata key for the candidate index within a turn.",
    )

    # === Reward Computation ===
    reward_from_score_diff: bool = Field(
        default=True,
        description="If True, compute trajectory reward as score difference (chosen - rejected). "
        "If False, use chosen score directly as reward.",
    )

    # === Validation ===
    require_multiple_candidates: bool = Field(
        default=True,
        description="If True, skip turns with only one candidate (no preference signal). "
        "If False, include single-candidate turns with a dummy rejected response.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "DPOTrajectoryBuilderConfig":
        """Validate configuration consistency."""
        if self.max_pairs_per_turn is not None and self.max_pairs_per_turn < 1:
            raise ValueError("max_pairs_per_turn must be at least 1 if specified")
        return self
