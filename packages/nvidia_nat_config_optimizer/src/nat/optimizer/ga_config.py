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
GA-specific optimizer config subtype.

Base types live in nat.data_models.optimizer (core). This package subclasses
BaseOptimizerConfig with the GA-specific nest (.prompt) and implementation-
specific fields (oracle feedback) that may not exist in other GA implementations.
"""

from typing import Literal

from pydantic import Field

from nat.data_models.optimizer import BaseOptimizerConfig
from nat.data_models.optimizer import PromptGAOptimizationConfig


class GAOptimizerConfig(BaseOptimizerConfig):
    """
    Optimizer config for this GA prompt implementation.

    Extends the shared base with .prompt (population, generations, init/recombine, etc.)
    and with oracle-feedback fields that are specific to this implementation—
    another GA might not use oracle feedback at all, so these live on the subtype.
    """
    prompt: PromptGAOptimizationConfig = Field(default_factory=PromptGAOptimizationConfig)

    # Oracle feedback: implementation-specific to this GA (inject failure reasoning into mutations).
    oracle_feedback_mode: Literal["never", "always", "failing_only", "adaptive"] = Field(
        description="When to inject failure reasoning into mutations.",
        default="never",
    )
    oracle_feedback_worst_n: int = Field(
        description="Number of worst-scoring items to extract reasoning from.",
        default=5,
        ge=1,
    )
    oracle_feedback_max_chars: int = Field(
        description="Maximum characters for oracle feedback in mutation prompt.",
        default=4000,
        ge=1,
    )
    oracle_feedback_fitness_threshold: float = Field(
        description="For 'failing_only' mode: normalized fitness threshold below which feedback is injected.",
        default=0.3,
        ge=0.0,
        le=1.0,
    )
    oracle_feedback_stagnation_generations: int = Field(
        description="For 'adaptive' mode: generations without improvement before enabling feedback.",
        default=3,
        ge=1,
    )
    oracle_feedback_fitness_variance_threshold: float = Field(
        description="For 'adaptive' mode: fitness variance threshold for collapse detection.",
        default=0.01,
        ge=0.0,
    )
    oracle_feedback_diversity_threshold: float = Field(
        description="For 'adaptive' mode: prompt duplication ratio threshold (0-1).",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
