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

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class PredictionMetrics(BaseModel):
    """Aggregated statistics for a single metric from profiler data."""

    sample_count: int = Field(default=0, description="Number of samples")
    mean: float = Field(default=0.0, description="Mean value")
    p50: float = Field(default=0.0, description="50th percentile (median)")
    p90: float = Field(default=0.0, description="90th percentile")
    p95: float = Field(default=0.0, description="95th percentile")


class LLMCallPrediction(BaseModel):
    """Predictions for an LLM call at a given position in the call hierarchy."""

    remaining_calls: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="How many more LLM calls are expected after this one",
    )
    interarrival_ms: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="Expected time in milliseconds until the next LLM call",
    )
    output_tokens: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="Expected output token count for this call",
    )


class PredictionTrieNode(BaseModel):
    """A node in the prediction trie representing a function in the call hierarchy."""

    name: str = Field(description="Function name at this level in the hierarchy")
    children: dict[str, PredictionTrieNode] = Field(
        default_factory=dict,
        description="Child nodes keyed by function name",
    )
    predictions_by_call_index: dict[int, LLMCallPrediction] = Field(
        default_factory=dict,
        description="Predictions keyed by call index (1-indexed)",
    )
    predictions_any_index: LLMCallPrediction | None = Field(
        default=None,
        description="Fallback predictions aggregated across all call indices",
    )


# Rebuild model to handle forward references
PredictionTrieNode.model_rebuild()
