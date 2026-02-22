# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from pydantic import ValidationError

from nat.data_models.optimizer import PromptGAOptimizationConfig


class TestOracleFeedbackConfig:
    """Tests for oracle feedback configuration fields."""

    def test_default_values(self):
        """Oracle feedback is disabled by default."""
        config = PromptGAOptimizationConfig()
        assert config.oracle_feedback_mode == "never"
        assert config.oracle_feedback_worst_n == 5
        assert config.oracle_feedback_max_chars == 4000
        assert config.oracle_feedback_fitness_threshold == 0.3
        assert config.oracle_feedback_stagnation_generations == 3
        assert config.oracle_feedback_fitness_variance_threshold == 0.01
        assert config.oracle_feedback_diversity_threshold == 0.5

    def test_valid_modes(self):
        """All valid feedback modes are accepted."""
        for mode in ["never", "always", "failing_only", "adaptive"]:
            config = PromptGAOptimizationConfig(oracle_feedback_mode=mode)
            assert config.oracle_feedback_mode == mode

    def test_invalid_mode_rejected(self):
        """Invalid feedback mode raises validation error."""
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_mode="invalid")

    def test_worst_n_must_be_positive(self):
        """oracle_feedback_worst_n must be >= 1."""
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_worst_n=0)

    def test_max_chars_must_be_positive(self):
        """oracle_feedback_max_chars must be >= 1."""
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_max_chars=0)

    def test_fitness_threshold_range(self):
        """oracle_feedback_fitness_threshold must be in [0, 1]."""
        PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=0.0)
        PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=1.0)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=-0.1)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_fitness_threshold=1.1)

    def test_diversity_threshold_range(self):
        """oracle_feedback_diversity_threshold must be in [0, 1]."""
        PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=0.0)
        PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=1.0)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=-0.1)
        with pytest.raises(ValidationError):
            PromptGAOptimizationConfig(oracle_feedback_diversity_threshold=1.1)
