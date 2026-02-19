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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.plugins.langchain.llm import dynamo_langchain
from nat.profiler.prediction_trie import save_prediction_trie
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup


@pytest.fixture(name="trie_file")
def fixture_trie_file():
    """Create a temporary trie file."""
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

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prediction_trie.json"
        save_prediction_trie(root, path, workflow_name="test")
        yield str(path)


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder."""
    return MagicMock(spec=Builder)


def test_dynamo_config_with_valid_trie_path(trie_file):
    """Test that DynamoModelConfig can be created with valid trie path."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path=trie_file,
    )

    assert config.prediction_trie_path == trie_file


def test_dynamo_config_with_nonexistent_trie_path():
    """Test that DynamoModelConfig accepts nonexistent path (validated at load time)."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path="/nonexistent/path/trie.json",
    )

    # Config creation should succeed; error happens at runtime
    assert config.prediction_trie_path == "/nonexistent/path/trie.json"


@patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
@patch("langchain_openai.ChatOpenAI")
async def test_dynamo_langchain_loads_trie_and_passes_to_client(mock_chat, mock_create_client, trie_file, mock_builder):
    """Test that dynamo_langchain loads trie from path and passes PredictionTrieLookup to httpx client."""
    mock_httpx_client = MagicMock()
    mock_httpx_client.aclose = AsyncMock()
    mock_create_client.return_value = mock_httpx_client

    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prefix_template="test-{uuid}",
        prediction_trie_path=trie_file,
    )

    async with dynamo_langchain(config, mock_builder):
        # Verify httpx client was created with prediction_lookup
        mock_create_client.assert_called_once()
        call_kwargs = mock_create_client.call_args.kwargs
        assert "prediction_lookup" in call_kwargs
        assert isinstance(call_kwargs["prediction_lookup"], PredictionTrieLookup)

    mock_httpx_client.aclose.assert_awaited_once()


@patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
@patch("langchain_openai.ChatOpenAI")
async def test_dynamo_langchain_handles_nonexistent_trie_gracefully(mock_chat, mock_create_client, mock_builder):
    """Test that dynamo_langchain logs warning and continues when trie file doesn't exist."""
    mock_httpx_client = MagicMock()
    mock_httpx_client.aclose = AsyncMock()
    mock_create_client.return_value = mock_httpx_client

    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prefix_template="test-{uuid}",
        prediction_trie_path="/nonexistent/path/trie.json",
    )

    # Should not raise an exception
    async with dynamo_langchain(config, mock_builder):
        # Verify httpx client was created with prediction_lookup=None
        mock_create_client.assert_called_once()
        call_kwargs = mock_create_client.call_args.kwargs
        assert call_kwargs["prediction_lookup"] is None

    mock_httpx_client.aclose.assert_awaited_once()


@patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
@patch("langchain_openai.ChatOpenAI")
async def test_dynamo_langchain_no_trie_path_means_no_lookup(mock_chat, mock_create_client, mock_builder):
    """Test that dynamo_langchain passes None when no trie path is configured."""
    mock_httpx_client = MagicMock()
    mock_httpx_client.aclose = AsyncMock()
    mock_create_client.return_value = mock_httpx_client

    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prefix_template="test-{uuid}",  # prediction_trie_path is None by default
    )

    async with dynamo_langchain(config, mock_builder):
        mock_create_client.assert_called_once()
        call_kwargs = mock_create_client.call_args.kwargs
        assert call_kwargs["prediction_lookup"] is None

    mock_httpx_client.aclose.assert_awaited_once()


@patch("nat.plugins.langchain.llm.create_httpx_client_with_dynamo_hooks")
@patch("langchain_openai.ChatOpenAI")
async def test_dynamo_langchain_handles_invalid_trie_file_gracefully(mock_chat, mock_create_client, mock_builder):
    """Test that dynamo_langchain logs warning and continues when trie file is invalid JSON."""
    mock_httpx_client = MagicMock()
    mock_httpx_client.aclose = AsyncMock()
    mock_create_client.return_value = mock_httpx_client

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json {{{")
        invalid_trie_path = f.name

    try:
        config = DynamoModelConfig(
            base_url="http://localhost:8000/v1",
            model_name="test-model",
            api_key="test-key",
            prefix_template="test-{uuid}",
            prediction_trie_path=invalid_trie_path,
        )

        # Should not raise an exception
        async with dynamo_langchain(config, mock_builder):
            # Verify httpx client was created with prediction_lookup=None
            mock_create_client.assert_called_once()
            call_kwargs = mock_create_client.call_args.kwargs
            assert call_kwargs["prediction_lookup"] is None

        mock_httpx_client.aclose.assert_awaited_once()
    finally:
        Path(invalid_trie_path).unlink(missing_ok=True)
