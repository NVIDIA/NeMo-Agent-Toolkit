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

import json
import tempfile
from pathlib import Path

import pytest

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.serialization import load_prediction_trie
from nat.profiler.prediction_trie.serialization import save_prediction_trie


@pytest.fixture(name="sample_trie")
def fixture_sample_trie() -> PredictionTrieNode:
    """Create a sample trie for testing serialization."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    child = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={1: prediction},
        predictions_any_index=prediction,
    )

    root = PredictionTrieNode(
        name="root",
        children={"react_agent": child},
        predictions_any_index=prediction,
    )

    return root


def test_save_and_load_trie(sample_trie):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prediction_trie.json"

        save_prediction_trie(sample_trie, path, workflow_name="test_workflow")

        loaded = load_prediction_trie(path)

        assert loaded.name == "root"
        assert "react_agent" in loaded.children
        assert loaded.children["react_agent"].predictions_by_call_index[1].remaining_calls.mean == 3.0


def test_saved_file_has_metadata(sample_trie):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prediction_trie.json"

        save_prediction_trie(sample_trie, path, workflow_name="test_workflow")

        with open(path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["workflow_name"] == "test_workflow"
        assert "generated_at" in data
        assert "root" in data
