# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
Tests for GAPromptOptimizationConfig oracle feedback fields.

Oracle feedback fields are typed directly on GAPromptOptimizationConfig
with proper defaults and validation.
"""

from nat.data_models.optimizer import GAPromptOptimizationConfig


class TestGAPromptOptimizationConfigOracleFeedback:
    """GAPromptOptimizationConfig has typed oracle feedback fields."""

    def test_oracle_feedback_fields_as_typed_attributes(self):
        """Oracle feedback fields are proper typed attributes with defaults."""
        config = GAPromptOptimizationConfig(
            oracle_feedback_mode="always",
            oracle_feedback_worst_n=3,
            oracle_feedback_max_chars=2000,
        )
        assert config.oracle_feedback_mode == "always"
        assert config.oracle_feedback_worst_n == 3
        assert config.oracle_feedback_max_chars == 2000

    def test_oracle_feedback_defaults(self):
        """Without oracle keys, defaults are applied."""
        config = GAPromptOptimizationConfig()
        assert config.oracle_feedback_mode == "never"
        assert config.oracle_feedback_worst_n == 5
        assert config.oracle_feedback_max_chars == 4000
        assert config.oracle_feedback_fitness_threshold == 0.3
        assert config.oracle_feedback_stagnation_generations == 3
        assert config.oracle_feedback_fitness_variance_threshold == 0.01
        assert config.oracle_feedback_diversity_threshold == 0.5
