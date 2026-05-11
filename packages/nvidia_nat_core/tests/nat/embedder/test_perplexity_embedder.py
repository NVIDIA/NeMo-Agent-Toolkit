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

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from nat.embedder.perplexity_embedder import PerplexityEmbedderModelConfig


def test_defaults():
    """Default config picks the small standard model and full dimensions."""
    config = PerplexityEmbedderModelConfig()
    assert config.type == "perplexity"
    assert config.model_name == "pplx-embed-v1-0.6b"
    assert config.base_url == "https://api.perplexity.ai/v1"
    assert config.dimensions is None
    assert config.batch_size == 64
    assert config.encoding_format == "base64_int8"


def test_accepts_supported_models():
    """All four model identifiers documented by the Perplexity API are accepted."""
    for model in (
        "pplx-embed-v1-0.6b",
        "pplx-embed-v1-4b",
        "pplx-embed-context-v1-0.6b",
        "pplx-embed-context-v1-4b",
    ):
        cfg = PerplexityEmbedderModelConfig(model_name=model)
        assert cfg.model_name == model


def test_rejects_unsupported_model():
    """Unsupported model identifiers raise a validation error."""
    with pytest.raises(ValidationError):
        PerplexityEmbedderModelConfig(model_name="text-embedding-3-small")


def test_dimensions_bounds():
    """Matryoshka ``dimensions`` is bounded to the documented range [128, 2560]."""
    PerplexityEmbedderModelConfig(model_name="pplx-embed-v1-4b", dimensions=128)
    PerplexityEmbedderModelConfig(model_name="pplx-embed-v1-4b", dimensions=2560)
    with pytest.raises(ValidationError):
        PerplexityEmbedderModelConfig(dimensions=64)
    with pytest.raises(ValidationError):
        PerplexityEmbedderModelConfig(dimensions=4096)


def test_batch_size_bounds():
    """``batch_size`` is bounded by Perplexity's documented 512-input-per-request cap."""
    PerplexityEmbedderModelConfig(batch_size=1)
    PerplexityEmbedderModelConfig(batch_size=512)
    with pytest.raises(ValidationError):
        PerplexityEmbedderModelConfig(batch_size=0)
    with pytest.raises(ValidationError):
        PerplexityEmbedderModelConfig(batch_size=1024)


def test_encoding_format_choices():
    """Only the two on-wire encoding formats supported by Perplexity are accepted."""
    PerplexityEmbedderModelConfig(encoding_format="base64_int8")
    PerplexityEmbedderModelConfig(encoding_format="base64_binary")
    with pytest.raises(ValidationError):
        # ``float`` is *not* supported by Perplexity's embeddings endpoint.
        PerplexityEmbedderModelConfig(encoding_format="float")


def test_api_key_secret_str():
    """``api_key`` is stored as a SecretStr-style field, not a plain string."""
    cfg = PerplexityEmbedderModelConfig(api_key="pplx-test-key")
    assert cfg.api_key is not None
    # Pydantic ``SecretStr`` masks the value in ``repr`` / ``str``.
    assert "pplx-test-key" not in repr(cfg.api_key)
    assert isinstance(cfg.api_key, SecretStr)


def test_model_alias_accepted():
    """The ``model`` alias is accepted in addition to ``model_name`` and round-trips."""
    cfg = PerplexityEmbedderModelConfig(model="pplx-embed-v1-4b")
    assert cfg.model_name == "pplx-embed-v1-4b"
