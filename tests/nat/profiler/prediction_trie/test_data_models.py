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

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


def test_prediction_metrics_creation():
    metrics = PredictionMetrics(sample_count=10, mean=5.0, p50=4.5, p90=8.0, p95=9.0)
    assert metrics.sample_count == 10
    assert metrics.mean == 5.0
    assert metrics.p50 == 4.5
    assert metrics.p90 == 8.0
    assert metrics.p95 == 9.0


def test_prediction_metrics_defaults():
    metrics = PredictionMetrics()
    assert metrics.sample_count == 0
    assert metrics.mean == 0.0


def test_llm_call_prediction_creation():
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=5, mean=3.0, p50=3.0, p90=5.0, p95=6.0),
        interarrival_ms=PredictionMetrics(sample_count=5, mean=500.0, p50=450.0, p90=800.0, p95=900.0),
        output_tokens=PredictionMetrics(sample_count=5, mean=150.0, p50=140.0, p90=250.0, p95=300.0),
    )
    assert prediction.remaining_calls.mean == 3.0
    assert prediction.interarrival_ms.mean == 500.0
    assert prediction.output_tokens.mean == 150.0


def test_llm_call_prediction_defaults():
    prediction = LLMCallPrediction()
    assert prediction.remaining_calls.sample_count == 0
    assert prediction.interarrival_ms.sample_count == 0
    assert prediction.output_tokens.sample_count == 0


def test_prediction_trie_node_creation():
    node = PredictionTrieNode(name="root")
    assert node.name == "root"
    assert node.children == {}
    assert node.predictions_by_call_index == {}
    assert node.predictions_any_index is None


def test_prediction_trie_node_with_children():
    child = PredictionTrieNode(name="react_agent")
    root = PredictionTrieNode(name="root", children={"react_agent": child})
    assert "react_agent" in root.children
    assert root.children["react_agent"].name == "react_agent"


def test_prediction_trie_node_with_predictions():
    prediction = LLMCallPrediction()
    node = PredictionTrieNode(
        name="agent",
        predictions_by_call_index={
            1: prediction, 2: prediction
        },
        predictions_any_index=prediction,
    )
    assert 1 in node.predictions_by_call_index
    assert 2 in node.predictions_by_call_index
    assert node.predictions_any_index is not None


def test_prediction_trie_node_nested_hierarchy():
    """Test a multi-level trie structure."""
    leaf = PredictionTrieNode(name="tool_call")
    middle = PredictionTrieNode(name="react_agent", children={"tool_call": leaf})
    root = PredictionTrieNode(name="workflow", children={"react_agent": middle})

    assert root.children["react_agent"].children["tool_call"].name == "tool_call"
