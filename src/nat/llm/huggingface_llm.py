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
"""
HuggingFace Transformers LLM Provider - Local in-process model execution.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.llm import LLMBaseConfig

logger = logging.getLogger(__name__)

# Global cache for loaded models
# Models remain cached for the provider's lifetime (not per-query!) to enable fast reuse:
# - During nat serve: Cached while server runs, cleaned up on shutdown
# - During nat red-team: Cached across all evaluation queries, cleaned up when complete
# - During nat run: Cached for single workflow execution, cleaned up when done
_model_cache = {}


class HuggingFaceConfig(LLMBaseConfig, name="huggingface"):
    """Configuration for HuggingFace LLM - loads model directly for local execution."""

    model_name: str = Field(description="HuggingFace model name (e.g. 'Qwen/Qwen3Guard-Gen-0.6B')")

    device: str = Field(default="auto", description="Device: 'cpu', 'cuda', 'cuda:0', or 'auto'")

    torch_dtype: str | None = Field(default="auto",
                                    description="Torch dtype: 'float16', 'bfloat16', 'float32', or 'auto'")

    max_new_tokens: int = Field(default=128, description="Maximum number of new tokens to generate")

    temperature: float = Field(default=0.0, description="Sampling temperature")

    trust_remote_code: bool = Field(default=False, description="Trust remote code when loading model")


class HuggingFaceModel(BaseChatModel):
    """LangChain-compatible wrapper for local HuggingFace models.

    This class inherits from BaseChatModel to provide proper LangChain integration
    for locally loaded HuggingFace Transformers models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Attributes (set during initialization)
    _model_name: str
    _config: HuggingFaceConfig
    _model: Any
    _tokenizer: Any
    _torch: Any

    def __init__(self, model_name: str, config: HuggingFaceConfig):
        """Initialize HuggingFace model wrapper.

        Args:
            model_name: Name of the loaded model
            config: Configuration for the model
        """
        # Get from cache
        if model_name not in _model_cache:
            raise ValueError(f"Model {model_name} not loaded in cache")

        cached = _model_cache[model_name]

        # Initialize parent
        super().__init__()

        # Set private attributes
        self._model_name = model_name
        self._config = config
        self._model = cached["model"]
        self._tokenizer = cached["tokenizer"]
        self._torch = cached["torch"]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def config(self) -> HuggingFaceConfig:
        """Return the model configuration."""
        return self._config

    @property
    def model(self):
        """Return the HuggingFace model."""
        return self._model

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def torch(self):
        """Return the torch module."""
        return self._torch

    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type."""
        return "huggingface"

    def _prepare_text(self, messages: list[BaseMessage] | list[dict] | str) -> str:
        """Convert messages to text using chat template or fallback.

        Args:
            messages: Input messages in various formats (BaseMessage list, dict list, or string)

        Returns:
            Formatted text string ready for tokenization
        """
        # Convert BaseMessage objects to dict format for template
        if isinstance(messages, list) and len(messages) > 0:
            # Handle LangChain BaseMessage objects
            if hasattr(messages[0], "type") and hasattr(messages[0], "content"):
                messages = [{
                    "role": msg.type, "content": msg.content
                } for msg in messages]  # type: ignore[attr-defined]

            # Try using chat template
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                # Fallback: just use the last message content
                logger.debug("Chat template application failed: %s, using fallback", e)
                last_msg = messages[-1]
                text = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
        else:
            text = str(messages)
        return text

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response synchronously (required by BaseChatModel).

        Args:
            messages: List of message objects.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager.
            kwargs: Additional generation parameters.

        Returns:
            ChatResult containing the generated response.
        """
        # Wrap async implementation
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Note: run_manager is sync but _agenerate expects async, so we don't pass it
        result = loop.run_until_complete(self._agenerate(messages, stop=stop, **kwargs))
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response asynchronously (called by BaseChatModel.ainvoke).

        Args:
            messages: List of message objects.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager.
            kwargs: Additional generation parameters.

        Returns:
            ChatResult containing the generated response.
        """
        # Convert messages to text
        text = self._prepare_text(messages)

        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        with self.torch.no_grad():
            generated_ids = self.model.generate(**model_inputs,
                                                max_new_tokens=self.config.max_new_tokens,
                                                temperature=self.config.temperature
                                                if self.config.temperature > 0 else None,
                                                do_sample=self.config.temperature > 0,
                                                pad_token_id=self.tokenizer.eos_token_id)

        # Decode (only new tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Return ChatResult (BaseChatModel format)
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        """Stream response tokens as they are generated (called by BaseChatModel.astream).

        Args:
            messages: List of message objects.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager.
            kwargs: Additional generation parameters.

        Yields:
            ChatGenerationChunk objects containing token chunks.
        """
        from langchain_core.outputs import ChatGenerationChunk

        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            # Fallback: if TextIteratorStreamer not available, yield full response
            logger.debug("TextIteratorStreamer not available, falling back to non-streaming")
            result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            # Convert AIMessage to AIMessageChunk for streaming
            full_message = result.generations[0].message
            chunk = AIMessageChunk(content=full_message.content)
            yield ChatGenerationChunk(message=chunk)
            return

        # Convert messages to text
        text = self._prepare_text(messages)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Create streamer for token-by-token generation
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        # Prepare generation kwargs
        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature if self.config.temperature > 0 else None,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # Start generation in background thread (model.generate is blocking)
        import asyncio
        import threading
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream tokens as they're generated
        try:
            for token_text in streamer:
                # Yield control to event loop
                await asyncio.sleep(0)

                # Return chunk in BaseChatModel format
                chunk = AIMessageChunk(content=token_text)
                yield ChatGenerationChunk(message=chunk)
        finally:
            # Ensure thread completes
            thread.join()

    def bind_tools(self, tools, **kwargs):
        """Bind tools to the LLM. Returns self to maintain fluent interface."""
        # HuggingFace models don't support tool calling, but we return self for compatibility
        return self

    def bind(self, **kwargs):
        """Bind additional parameters to the LLM. Returns self to maintain fluent interface."""
        # HuggingFace models don't support parameter binding, but we return self for compatibility
        return self


async def _cleanup_model(model_name: str) -> None:
    """Clean up a loaded model and free GPU memory.

    Args:
        model_name: Name of the model to clean up.
    """
    try:
        if model_name in _model_cache:
            cached = _model_cache[model_name]

            # Move model to CPU to free GPU memory
            if "model" in cached:
                cached["model"].to("cpu")
                del cached["model"]

            # Clear CUDA cache if available
            if "torch" in cached and hasattr(cached["torch"].cuda, "empty_cache"):
                cached["torch"].cuda.empty_cache()

            # Remove from cache
            del _model_cache[model_name]

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
    except ImportError as err:
        raise ImportError(
            "transformers and torch required. Install: pip install transformers torch accelerate") from err

    # Load model if not cached
    if config.model_name not in _model_cache:
        logger.debug("Loading model from HuggingFace: %s", config.model_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                     torch_dtype=config.torch_dtype,
                                                     device_map=config.device,
                                                     trust_remote_code=config.trust_remote_code)

        # Cache it
        _model_cache[config.model_name] = {"model": model, "tokenizer": tokenizer, "torch": torch}

        logger.debug("Model loaded: %s on device: %s", config.model_name, config.device)
    else:
        logger.debug("Using cached model: %s", config.model_name)

    try:
        yield LLMProviderInfo(config=config, description=f"HuggingFace model: {config.model_name}")
    finally:
        # Cleanup when workflow/application shuts down
        await _cleanup_model(config.model_name)


def get_huggingface_model(model_name: str, config: HuggingFaceConfig):
    """Create a HuggingFace model wrapper for a loaded model.

    Args:
        model_name: Name of the model to retrieve.
        config: Configuration for the model wrapper.

    Returns:
        HuggingFaceModel instance or None if model not loaded.
    """
    if model_name in _model_cache:
        return HuggingFaceModel(model_name, config)
    return None
