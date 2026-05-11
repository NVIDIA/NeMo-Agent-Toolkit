# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from nat.embedder.perplexity_embedder import PerplexityEmbedderModelConfig
from nat.plugins.langchain.embedder import perplexity_langchain
from nat.plugins.langchain.perplexity_embeddings_client import PerplexityEmbeddings
from nat.plugins.langchain.perplexity_embeddings_client import _decode_binary
from nat.plugins.langchain.perplexity_embeddings_client import _decode_int8
from nat.plugins.langchain.perplexity_embeddings_client import _default_integration_header


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def _encode_int8(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int8).tobytes()).decode()


def _make_response(status_code: int, payload: dict) -> httpx.Response:
    """Build an ``httpx.Response`` with a bound dummy request so ``raise_for_status`` works in tests."""
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.perplexity.ai/v1/embeddings"),
    )


def _encode_binary(bits: list[int]) -> str:
    packed = np.packbits(np.array(bits, dtype=np.uint8), bitorder="little")
    return base64.b64encode(packed.tobytes()).decode()


def test_decode_int8_round_trip():
    payload = _encode_int8([-128, -1, 0, 1, 127])
    decoded = _decode_int8(payload)
    assert decoded == [-128.0, -1.0, 0.0, 1.0, 127.0]


def test_decode_binary_round_trip():
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    payload = _encode_binary(bits)
    decoded = _decode_binary(payload)
    assert decoded == [float(b) for b in bits]


def test_default_integration_header_uses_nemo_slug():
    header = _default_integration_header()
    assert header.startswith("nemo-agent-toolkit/")


# ---------------------------------------------------------------------------
# PerplexityEmbeddings client
# ---------------------------------------------------------------------------


class TestPerplexityEmbeddingsClient:

    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            PerplexityEmbeddings(api_key="")

    def test_rejects_unsupported_encoding(self):
        with pytest.raises(ValueError):
            PerplexityEmbeddings(api_key="pplx-x", encoding_format="float")

    def test_batches_inputs_and_decodes_int8(self):
        """The client should split inputs into batches of ``batch_size`` and decode int8 payloads."""
        client = PerplexityEmbeddings(api_key="pplx-x", batch_size=2)

        responses = [
            _make_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": _encode_int8([1, 2])},
                        {"object": "embedding", "index": 1, "embedding": _encode_int8([3, 4])},
                    ],
                    "model": "pplx-embed-v1-0.6b",
                },
            ),
            _make_response(
                200,
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": _encode_int8([5, 6])}],
                    "model": "pplx-embed-v1-0.6b",
                },
            ),
        ]

        with patch("httpx.Client") as mock_client_cls:
            instance = MagicMock()
            instance.__enter__.return_value = instance
            instance.post.side_effect = responses
            mock_client_cls.return_value = instance

            vectors = client.embed_documents(["a", "b", "c"])

        assert vectors == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        # Two batches of size 2 (last one is size 1).
        assert instance.post.call_count == 2

    def test_request_body_includes_dimensions_and_attribution_header(self):
        client = PerplexityEmbeddings(api_key="pplx-x", dimensions=256, integration_header="test-suite/1.2.3")

        with patch("httpx.Client") as mock_client_cls:
            instance = MagicMock()
            instance.__enter__.return_value = instance
            instance.post.return_value = _make_response(
                200,
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": _encode_int8([0])}],
                    "model": "pplx-embed-v1-0.6b",
                },
            )
            mock_client_cls.return_value = instance

            client.embed_query("hello")

        call = instance.post.call_args
        assert call.kwargs["json"]["dimensions"] == 256
        assert call.kwargs["json"]["encoding_format"] == "base64_int8"
        assert call.kwargs["json"]["model"] == "pplx-embed-v1-0.6b"
        assert call.kwargs["headers"]["X-Pplx-Integration"] == "test-suite/1.2.3"
        assert call.kwargs["headers"]["Authorization"] == "Bearer pplx-x"
        assert call.args[0].endswith("/v1/embeddings")

    def test_non_retriable_status_raises(self):
        client = PerplexityEmbeddings(api_key="pplx-x", max_retries=3)
        with patch("httpx.Client") as mock_client_cls:
            instance = MagicMock()
            instance.__enter__.return_value = instance
            instance.post.return_value = _make_response(401, {"error": "unauthorized"})
            mock_client_cls.return_value = instance

            with pytest.raises(httpx.HTTPStatusError):
                client.embed_query("hello")

            # 401 short-circuits — exactly one POST attempt.
            assert instance.post.call_count == 1

    async def test_async_embed_query(self):
        client = PerplexityEmbeddings(api_key="pplx-x")

        async_response = _make_response(
            200,
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": _encode_int8([7, 8, 9])}],
                "model": "pplx-embed-v1-0.6b",
            },
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            instance = MagicMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.post = AsyncMock(return_value=async_response)
            mock_client_cls.return_value = instance

            vector = await client.aembed_query("hi")

        assert vector == [7.0, 8.0, 9.0]


# ---------------------------------------------------------------------------
# perplexity_langchain registration
# ---------------------------------------------------------------------------


class TestPerplexityLangChainRegistration:

    async def test_requires_api_key_in_env_or_config(self, monkeypatch, mock_builder):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        cfg = PerplexityEmbedderModelConfig()
        with pytest.raises(ValueError, match="non-empty API key"):
            async with perplexity_langchain(cfg, mock_builder):
                pass

    async def test_uses_environment_api_key(self, monkeypatch, mock_builder):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
        cfg = PerplexityEmbedderModelConfig()
        async with perplexity_langchain(cfg, mock_builder) as client:
            assert hasattr(client, "embed_documents")
            assert hasattr(client, "embed_query")
