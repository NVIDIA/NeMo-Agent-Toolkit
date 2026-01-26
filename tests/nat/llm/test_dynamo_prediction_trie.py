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

import tempfile
from pathlib import Path

import pytest

from nat.llm.dynamo_llm import DynamoModelConfig
from nat.profiler.prediction_trie import PredictionTrieNode
from nat.profiler.prediction_trie import save_prediction_trie
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics


@pytest.fixture(name="trie_file")
def fixture_trie_file() -> Path:
    """Create a temporary trie file for testing."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    root = PredictionTrieNode(
        name="root",
        predictions_by_call_index={1: prediction},
        predictions_any_index=prediction,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = Path(f.name)

    save_prediction_trie(root, path)
    yield path
    path.unlink(missing_ok=True)


def test_dynamo_config_with_trie_path(trie_file):
    """Test that DynamoModelConfig accepts prediction_trie_path."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path=str(trie_file),
    )

    assert config.prediction_trie_path == str(trie_file)
    assert "prediction_trie_path" in DynamoModelConfig.get_dynamo_field_names()


def test_dynamo_config_without_trie_path():
    """Test that DynamoModelConfig works without prediction_trie_path."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000",
        model_name="test-model",
        api_key="test-key",
    )

    assert config.prediction_trie_path is None


def test_dynamo_field_names_excludes_trie_path():
    """Test that prediction_trie_path is excluded from OpenAI client kwargs."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path="/path/to/trie.json",
    )

    # Simulate what would be passed to an OpenAI client
    exclude_fields = {"type", "thinking", *DynamoModelConfig.get_dynamo_field_names()}
    config_dict = config.model_dump(exclude=exclude_fields, exclude_none=True)

    # prediction_trie_path should not be in the config dict
    assert "prediction_trie_path" not in config_dict
