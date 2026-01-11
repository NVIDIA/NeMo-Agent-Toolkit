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
"""HuggingFace Transformers LLM Provider - Local in-process model execution."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.llm import LLMBaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheEntry:
    model: Any
    tokenizer: Any
    torch: Any


class ModelCache:
    """Singleton cache for loaded HuggingFace models.

    Models remain cached for the provider's lifetime (not per-query!) to enable fast reuse:
    - During nat serve: Cached while server runs, cleaned up on shutdown
    - During nat red-team: Cached across all evaluation queries, cleaned up when complete
    - During nat run: Cached for single workflow execution, cleaned up when done
    """

    _instance: "ModelCache | None" = None
    _cache: dict[str, ModelCacheEntry]

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get(self, model_name: str) -> ModelCacheEntry | None:
        """Return cached model data or None if not loaded."""
        return self._cache.get(model_name)

    def set(self, model_name: str, data: ModelCacheEntry) -> None:
        """Cache model data."""
        self._cache[model_name] = data

    def remove(self, model_name: str) -> None:
        """Remove model from cache."""
        self._cache.pop(model_name, None)

    def __contains__(self, model_name: str) -> bool:
        """Check if model is cached."""
        return model_name in self._cache


class HuggingFaceConfig(LLMBaseConfig, name="huggingface"):
    """Configuration for HuggingFace LLM - loads model directly for local execution."""

    model_name: str = Field(description="HuggingFace model name (e.g. 'Qwen/Qwen3Guard-Gen-0.6B')")

    device: str = Field(default="auto", description="Device: 'cpu', 'cuda', 'cuda:0', or 'auto'")

    dtype: str | None = Field(default="auto", description="Torch dtype: 'float16', 'bfloat16', 'float32', or 'auto'")

    max_new_tokens: int = Field(default=128, description="Maximum number of new tokens to generate")

    temperature: float = Field(default=0.0,
                               description="Sampling temperature (0 = deterministic greedy, > 0 = sampling enabled)")

    trust_remote_code: bool = Field(default=False, description="Trust remote code when loading model")


def get_cached_model(model_name: str) -> ModelCacheEntry | None:
    """Return cached model data (model, tokenizer, torch) or None if not loaded."""
    return ModelCache().get(model_name)


async def _cleanup_model(model_name: str) -> None:
    """Clean up a loaded model and free GPU memory.

    Args:
        model_name: Name of the model to clean up.
    """
    try:
        cache = ModelCache()
        cached = cache.get(model_name)

        if cached is not None:
            # Move model to CPU to free GPU memory
            if cached.model:
                cached.model.to("cpu")
                cached.model = None

            # Clear CUDA cache if available
            if cached.torch and hasattr(cached.torch.cuda, "empty_cache"):
                cached.torch.cuda.empty_cache()
                cached.torch = None

            cached.tokenizer = None

            # Remove from cache
            cache.remove(model_name)

            logger.debug("Model cleaned up: %s", model_name)
    except Exception:
        logger.exception("Error cleaning up HuggingFace model '%s'", model_name)


@register_llm_provider(config_type=HuggingFaceConfig)
async def huggingface_provider(
        config: HuggingFaceConfig,
        builder: Builder,  # noqa: ARG001 - kept for provider interface, currently unused
) -> AsyncIterator[LLMProviderInfo]:
    """HuggingFace model provider - loads models locally for in-process execution.

    Args:
        config: Configuration for the HuggingFace model.
        builder: The NAT builder instance.

    Yields:
        LLMProviderInfo: Provider information for the loaded model.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError("HuggingFace dependencies not installed. \n"
                         "Install with:\n"
                         "  `pip install nvidia-nat[huggingface]` (package) or\n"
                         "  `uv pip install -e '.[huggingface]'` (source)\n")

    cache = ModelCache()

    # Load model if not cached
    if config.model_name not in cache:
        logger.debug("Loading model from HuggingFace: %s", config.model_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                     dtype=config.dtype,
                                                     device_map=config.device,
                                                     trust_remote_code=config.trust_remote_code)

        # Cache it
        cache.set(config.model_name, ModelCacheEntry(model=model, tokenizer=tokenizer, torch=torch))

        logger.debug("Model loaded: %s on device: %s", config.model_name, config.device)
    else:
        logger.debug("Using cached model: %s", config.model_name)

    try:
        yield LLMProviderInfo(config=config, description=f"HuggingFace model: {config.model_name}")
    finally:
        # Cleanup when workflow/application shuts down
        await _cleanup_model(config.model_name)
