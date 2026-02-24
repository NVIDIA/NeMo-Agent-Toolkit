# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
Tests for PromptGAOptimizationConfig and extra (oracle) fields.

Oracle feedback is implementation-specific; it is not part of the shared
PromptGAOptimizationConfig contract. Extra fields (e.g. oracle_feedback_mode)
are allowed and stored in model_extra so YAML can still specify them.
Validation of oracle_* is done by GAOptimizerConfig in nvidia_nat_config_optimizer.
"""

from nat.data_models.optimizer import PromptGAOptimizationConfig


class TestPromptGAOptimizationConfigExtra:
    """PromptGAOptimizationConfig accepts oracle_* as extra and stores in model_extra."""

    def test_accepts_oracle_extra_fields(self):
        """Oracle feedback keys are accepted as extra and stored in model_extra."""
        config = PromptGAOptimizationConfig(
            oracle_feedback_mode="always",
            oracle_feedback_worst_n=3,
            oracle_feedback_max_chars=2000,
        )
        assert config.model_extra is not None
        assert config.model_extra.get("oracle_feedback_mode") == "always"
        assert config.model_extra.get("oracle_feedback_worst_n") == 3
        assert config.model_extra.get("oracle_feedback_max_chars") == 2000

    def test_no_oracle_extra_by_default(self):
        """Without oracle keys, model_extra is empty or absent."""
        config = PromptGAOptimizationConfig()
        assert getattr(config, "model_extra", None) is None or config.model_extra == {}
