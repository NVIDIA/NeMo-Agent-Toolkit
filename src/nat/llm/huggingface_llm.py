# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

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
    
    model_name: str = Field(
        description="HuggingFace model name (e.g. 'Qwen/Qwen3Guard-Gen-0.6B')"
    )
    
    device: str = Field(
        default="auto",
        description="Device: 'cpu', 'cuda', 'cuda:0', or 'auto'"
    )
    
    torch_dtype: str | None = Field(
        default="auto",
        description="Torch dtype: 'float16', 'bfloat16', 'float32', or 'auto'"
    )
    
    max_new_tokens: int = Field(
        default=128,
        description="Maximum number of new tokens to generate"
    )
    
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature"
    )
    
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code when loading model"
    )


class HuggingFaceModel:
    """Wrapper that provides LangChain-compatible interface for local HuggingFace models."""
    
    def __init__(self, model_name: str, config: HuggingFaceConfig):
        self.model_name = model_name
        self.config = config
        
        # Get from cache
        if model_name not in _model_cache:
            raise ValueError(f"Model {model_name} not loaded in cache")
        
        cached = _model_cache[model_name]
        self.model = cached["model"]
        self.tokenizer = cached["tokenizer"]
        self.torch = cached["torch"]
    
    def _prepare_text(self, messages):
        """Convert messages to text using chat template or fallback."""
        if isinstance(messages, list) and len(messages) > 0:
            # Try using chat template
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                # Fallback: just use the last message content
                logger.debug("Chat template application failed: %s, using fallback", e)
                last_msg = messages[-1]
                text = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
        else:
            text = str(messages)
        return text
    
    def invoke(self, messages, **kwargs):
        """Synchronous invoke - wraps async version."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.ainvoke(messages, **kwargs))
    
    async def ainvoke(self, messages, **kwargs):
        """Generate response - matches LangChain interface."""
        from langchain_core.messages import AIMessage
        
        # Convert messages to text
        text = self._prepare_text(messages)
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (only new tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Return AIMessage (matches LangChain interface)
        return AIMessage(content=content)
    
    def stream(self, messages, **kwargs):
        """Synchronous stream - wraps async version."""
        import asyncio
        from collections.abc import Iterator
        
        async def _collect():
            chunks = []
            async for chunk in self.astream(messages, **kwargs):
                chunks.append(chunk)
            return chunks
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        chunks = loop.run_until_complete(_collect())
        for chunk in chunks:
            yield chunk
    
    async def astream(self, messages, **kwargs):
        """Stream response token by token - matches LangChain streaming interface."""
        from langchain_core.messages import AIMessageChunk
        import asyncio
        from threading import Thread
        
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            # Fallback: if TextIteratorStreamer not available, yield full response
            logger.debug("TextIteratorStreamer not available, falling back to non-streaming")
            response = await self.ainvoke(messages)
            yield AIMessageChunk(content=response.content)
            return
        
        # Convert messages to text
        text = self._prepare_text(messages)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Create streamer for token-by-token generation
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True
        )
        
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
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens as they're generated
        try:
            for token_text in streamer:
                # Yield control to event loop
                await asyncio.sleep(0)
                
                # Return chunk in LangChain format
                yield AIMessageChunk(content=token_text)
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
async def huggingface_provider(config: HuggingFaceConfig, builder: Builder):
    """HuggingFace model provider - loads models locally for in-process execution.
    
    Args:
        config: Configuration for the HuggingFace model.
        builder: The NAT builder instance.
        
    Yields:
        LLMProviderInfo: Provider information for the loaded model.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        raise ImportError(
            "transformers and torch required. Install: pip install transformers torch accelerate"
        ) from err
    
    # Load model if not cached
    if config.model_name not in _model_cache:
        logger.debug("Loading model from HuggingFace: %s", config.model_name)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device,
            trust_remote_code=config.trust_remote_code
        )
        
        # Cache it
        _model_cache[config.model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "torch": torch
        }
        
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
