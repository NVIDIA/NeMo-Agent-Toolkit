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
"""LangChain ``Embeddings`` client for the Perplexity Embeddings API.

Perplexity's embeddings endpoint (``POST /v1/embeddings``) returns base64-encoded
quantized values (``base64_int8`` or ``base64_binary``) rather than the JSON float
arrays returned by OpenAI-compatible providers. This module provides a thin
LangChain :class:`~langchain_core.embeddings.Embeddings` implementation that
performs the decoding so the resulting vectors plug into the standard NAT
retriever/RAG stack.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import Iterable

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def _decode_int8(payload: str) -> list[float]:
    """Decode a ``base64_int8`` embedding into a float32 list.

    Args:
        payload: Base64-encoded signed int8 buffer returned by Perplexity.

    Returns:
        The decoded vector as a list of Python floats.
    """
    return np.frombuffer(base64.b64decode(payload), dtype=np.int8).astype(np.float32).tolist()


def _decode_binary(payload: str) -> list[float]:
    """Decode a ``base64_binary`` embedding into a 0/1 float list.

    Args:
        payload: Base64-encoded packed bits (LSB first, 1 bit per dimension).

    Returns:
        The unpacked bit vector as a list of Python floats (0.0 or 1.0).
    """
    packed = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little").astype(np.float32).tolist()


class PerplexityEmbeddings(Embeddings):
    """LangChain ``Embeddings`` client for the Perplexity Embeddings API.

    The client batches inputs (``batch_size`` per request, default 64), decodes the
    base64 payload locally, and forwards the ``X-Pplx-Integration`` attribution
    header so Perplexity can identify NeMo Agent Toolkit traffic.

    Args:
        api_key: Perplexity API key. Required.
        base_url: Base URL for the Perplexity API. Defaults to ``https://api.perplexity.ai/v1``.
        model: Embedding model identifier (e.g. ``pplx-embed-v1-0.6b``).
        dimensions: Optional Matryoshka output dimension.
        encoding_format: ``base64_int8`` (default) or ``base64_binary``.
        batch_size: Maximum inputs per request (1–512). Defaults to 64.
        max_retries: Number of retry attempts on transient failures. Defaults to 3.
        verify_ssl: Whether to verify TLS certificates. Defaults to True.
        integration_header: Optional ``X-Pplx-Integration`` header value. Defaults
            to ``"nemo-agent-toolkit/<version>"`` resolved at instantiation time.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.perplexity.ai/v1",
        model: str = "pplx-embed-v1-0.6b",
        dimensions: int | None = None,
        encoding_format: str = "base64_int8",
        batch_size: int = 64,
        max_retries: int = 3,
        verify_ssl: bool = True,
        integration_header: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "PerplexityEmbeddings requires a non-empty api_key. "
                "Set the PERPLEXITY_API_KEY environment variable or pass api_key explicitly."
            )
        if encoding_format not in ("base64_int8", "base64_binary"):
            raise ValueError(
                f"encoding_format must be 'base64_int8' or 'base64_binary', got {encoding_format!r}."
            )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._encoding_format = encoding_format
        self._batch_size = max(1, min(int(batch_size), 512))
        self._max_retries = max(1, int(max_retries))
        self._verify_ssl = verify_ssl
        self._integration_header = integration_header or _default_integration_header()

    # ------------------------------------------------------------------
    # LangChain interface
    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:  # type: ignore[override]
        results = self._embed([text])
        return results[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        return await self._aembed(texts)

    async def aembed_query(self, text: str) -> list[float]:  # type: ignore[override]
        results = await self._aembed([text])
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _endpoint(self) -> str:
        return f"{self._base_url}/embeddings"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Pplx-Integration": self._integration_header,
        }

    def _build_body(self, inputs: list[str]) -> dict:
        body: dict = {
            "input": inputs,
            "model": self._model,
            "encoding_format": self._encoding_format,
        }
        if self._dimensions is not None:
            body["dimensions"] = self._dimensions
        return body

    def _decode_response(self, payload: dict) -> list[list[float]]:
        data: Iterable[dict] = payload.get("data") or []
        decoder = _decode_int8 if self._encoding_format == "base64_int8" else _decode_binary
        # Preserve input order via the ``index`` field, which the API echoes back.
        decoded: list[list[float]] = [None] * len(list(data))  # type: ignore[list-item]
        items = list(payload.get("data") or [])
        decoded = [decoder(item.get("embedding", "")) for item in items]
        return decoded

    def _batches(self, texts: list[str]) -> Iterable[list[str]]:
        for start in range(0, len(texts), self._batch_size):
            yield texts[start:start + self._batch_size]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        with httpx.Client(verify=self._verify_ssl, timeout=60.0) as client:
            for batch in self._batches(texts):
                payload = self._post_with_retry_sync(client, batch)
                embeddings.extend(self._decode_response(payload))
        return embeddings

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(verify=self._verify_ssl, timeout=60.0) as client:
            for batch in self._batches(texts):
                payload = await self._post_with_retry_async(client, batch)
                embeddings.extend(self._decode_response(payload))
        return embeddings

    def _post_with_retry_sync(self, client: httpx.Client, batch: list[str]) -> dict:
        import time

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = client.post(self._endpoint(), headers=self._headers(), json=self._build_body(batch))
                if response.status_code in {401, 403, 404}:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code in {401, 403, 404} or attempt == self._max_retries - 1:
                    raise
                logger.warning(
                    "Perplexity embeddings attempt %d/%d failed with status %d",
                    attempt + 1,
                    self._max_retries,
                    exc.response.status_code,
                )
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == self._max_retries - 1:
                    raise
                logger.warning(
                    "Perplexity embeddings attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
            time.sleep(2**attempt)
        raise RuntimeError("Perplexity embeddings request failed") from last_error

    async def _post_with_retry_async(self, client: httpx.AsyncClient, batch: list[str]) -> dict:
        import asyncio

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await client.post(
                    self._endpoint(), headers=self._headers(), json=self._build_body(batch)
                )
                if response.status_code in {401, 403, 404}:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code in {401, 403, 404} or attempt == self._max_retries - 1:
                    raise
                logger.warning(
                    "Perplexity embeddings attempt %d/%d failed with status %d",
                    attempt + 1,
                    self._max_retries,
                    exc.response.status_code,
                )
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == self._max_retries - 1:
                    raise
                logger.warning(
                    "Perplexity embeddings attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
            await asyncio.sleep(2**attempt)
        raise RuntimeError("Perplexity embeddings request failed") from last_error


def _default_integration_header() -> str:
    """Return the default ``X-Pplx-Integration`` header value for outbound requests."""
    from importlib import metadata

    try:
        package_version = metadata.version("nvidia-nat")
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    return f"nemo-agent-toolkit/{package_version}"
