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

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup


@pytest.fixture(name="sample_trie")
def fixture_sample_trie() -> PredictionTrieNode:
    """Create a sample trie for testing lookups."""
    prediction_1 = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )
    prediction_2 = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=2.0, p50=2.0, p90=3.0, p95=4.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=400.0, p50=380.0, p90=600.0, p95=700.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=200.0, p50=190.0, p90=280.0, p95=320.0),
    )
    aggregated = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=20, mean=2.5, p50=2.5, p90=3.5, p95=4.5),
        interarrival_ms=PredictionMetrics(sample_count=20, mean=450.0, p50=415.0, p90=650.0, p95=750.0),
        output_tokens=PredictionMetrics(sample_count=20, mean=175.0, p50=165.0, p90=240.0, p95=285.0),
    )

    agent_node = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={
            1: prediction_1, 2: prediction_2
        },
        predictions_any_index=aggregated,
    )

    workflow_node = PredictionTrieNode(
        name="my_workflow",
        children={"react_agent": agent_node},
        predictions_any_index=aggregated,
    )

    root = PredictionTrieNode(
        name="root",
        children={"my_workflow": workflow_node},
        predictions_any_index=aggregated,
    )

    return root


def test_lookup_exact_match(sample_trie):
    lookup = PredictionTrieLookup(sample_trie)
    result = lookup.find(path=["my_workflow", "react_agent"], call_index=1)

    assert result is not None
    assert result.remaining_calls.mean == 3.0
    assert result.output_tokens.mean == 150.0


def test_lookup_partial_path_match(sample_trie):
    """When exact path doesn't exist, fall back to closest ancestor."""
    lookup = PredictionTrieLookup(sample_trie)
    # "unknown_tool" doesn't exist, should fall back to react_agent's aggregated
    result = lookup.find(path=["my_workflow", "react_agent", "unknown_tool"], call_index=1)

    assert result is not None
    # Should get react_agent's call_index=1 prediction
    assert result.remaining_calls.mean == 3.0


def test_lookup_unknown_call_index_fallback(sample_trie):
    """When call_index doesn't exist, fall back to aggregated."""
    lookup = PredictionTrieLookup(sample_trie)
    result = lookup.find(path=["my_workflow", "react_agent"], call_index=99)

    assert result is not None
    # Should fall back to predictions_any_index
    assert result.remaining_calls.mean == 2.5


def test_lookup_no_match_returns_root_aggregated(sample_trie):
    """When nothing matches, return root's aggregated."""
    lookup = PredictionTrieLookup(sample_trie)
    result = lookup.find(path=["completely_unknown"], call_index=1)

    assert result is not None
    # Should return root's aggregated prediction
    assert result.remaining_calls.mean == 2.5
