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
"""Tests for DPO Trajectory Builder configuration."""

import pytest
from pydantic import ValidationError

from nat.plugins.customizer.dpo.config import DPOTrajectoryBuilderConfig


class TestDPOTrajectoryBuilderConfig:
    """Tests for DPOTrajectoryBuilderConfig validation and defaults."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DPOTrajectoryBuilderConfig()

        assert config.custom_step_name == "dpo_candidate_move"
        assert config.exhaustive_pairs is True
        assert config.min_score_diff == 0.0
        assert config.max_pairs_per_turn is None
        assert config.turn_id_key == "turn_id"
        assert config.score_key == "score"
        assert config.prompt_key == "prompt"
        assert config.response_key == "raw_llm_response"
        assert config.candidate_index_key == "candidate_index"
        assert config.reward_from_score_diff is True
        assert config.require_multiple_candidates is True
        assert config.include_system_prompt is False
        assert config.system_prompt_key == "system_prompt"

    def test_custom_step_name(self):
        """Test custom step name configuration."""
        config = DPOTrajectoryBuilderConfig(custom_step_name="my_custom_step")
        assert config.custom_step_name == "my_custom_step"

    def test_exhaustive_pairs_false(self):
        """Test disabling exhaustive pair generation."""
        config = DPOTrajectoryBuilderConfig(exhaustive_pairs=False)
        assert config.exhaustive_pairs is False

    def test_min_score_diff_positive(self):
        """Test positive minimum score difference."""
        config = DPOTrajectoryBuilderConfig(min_score_diff=0.1)
        assert config.min_score_diff == 0.1

    def test_min_score_diff_negative_fails(self):
        """Test that negative min_score_diff raises validation error."""
        with pytest.raises(ValidationError):
            DPOTrajectoryBuilderConfig(min_score_diff=-0.1)

    def test_max_pairs_per_turn_valid(self):
        """Test valid max_pairs_per_turn values."""
        config = DPOTrajectoryBuilderConfig(max_pairs_per_turn=5)
        assert config.max_pairs_per_turn == 5

        config = DPOTrajectoryBuilderConfig(max_pairs_per_turn=1)
        assert config.max_pairs_per_turn == 1

    def test_max_pairs_per_turn_zero_fails(self):
        """Test that zero max_pairs_per_turn raises validation error."""
        with pytest.raises(ValidationError):
            DPOTrajectoryBuilderConfig(max_pairs_per_turn=0)

    def test_max_pairs_per_turn_none(self):
        """Test that None max_pairs_per_turn is allowed (unlimited)."""
        config = DPOTrajectoryBuilderConfig(max_pairs_per_turn=None)
        assert config.max_pairs_per_turn is None

    def test_custom_field_mappings(self):
        """Test custom field mappings for different workflows."""
        config = DPOTrajectoryBuilderConfig(
            turn_id_key="game_turn",
            score_key="evaluation_score",
            prompt_key="input_text",
            response_key="output_text",
            candidate_index_key="candidate_num",
        )

        assert config.turn_id_key == "game_turn"
        assert config.score_key == "evaluation_score"
        assert config.prompt_key == "input_text"
        assert config.response_key == "output_text"
        assert config.candidate_index_key == "candidate_num"

    def test_reward_from_score_diff_false(self):
        """Test using chosen score as reward instead of diff."""
        config = DPOTrajectoryBuilderConfig(reward_from_score_diff=False)
        assert config.reward_from_score_diff is False

    def test_require_multiple_candidates_false(self):
        """Test allowing single candidate turns."""
        config = DPOTrajectoryBuilderConfig(require_multiple_candidates=False)
        assert config.require_multiple_candidates is False

    def test_include_system_prompt(self):
        """Test system prompt inclusion configuration."""
        config = DPOTrajectoryBuilderConfig(
            include_system_prompt=True,
            system_prompt_key="custom_system_prompt",
        )
        assert config.include_system_prompt is True
        assert config.system_prompt_key == "custom_system_prompt"

    def test_config_name(self):
        """Test that config is registered with correct name."""
        # The name is stored in _typed_model_name class variable
        assert DPOTrajectoryBuilderConfig._typed_model_name == "dpo_traj_builder"

    def test_model_validator(self):
        """Test model validator runs successfully."""
        # Valid config should pass validation
        config = DPOTrajectoryBuilderConfig(max_pairs_per_turn=10)
        assert config.max_pairs_per_turn == 10

    def test_full_configuration(self):
        """Test a complete configuration with all options set."""
        config = DPOTrajectoryBuilderConfig(
            custom_step_name="custom_dpo_step",
            exhaustive_pairs=False,
            min_score_diff=0.05,
            max_pairs_per_turn=3,
            include_system_prompt=True,
            system_prompt_key="sys_prompt",
            turn_id_key="turn",
            score_key="eval_score",
            prompt_key="input",
            response_key="output",
            candidate_index_key="idx",
            reward_from_score_diff=False,
            require_multiple_candidates=False,
        )

        assert config.custom_step_name == "custom_dpo_step"
        assert config.exhaustive_pairs is False
        assert config.min_score_diff == 0.05
        assert config.max_pairs_per_turn == 3
        assert config.include_system_prompt is True
        assert config.system_prompt_key == "sys_prompt"
        assert config.turn_id_key == "turn"
        assert config.score_key == "eval_score"
        assert config.prompt_key == "input"
        assert config.response_key == "output"
        assert config.candidate_index_key == "idx"
        assert config.reward_from_score_diff is False
        assert config.require_multiple_candidates is False
