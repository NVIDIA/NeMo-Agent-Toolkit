# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple HuggingFace Transformers LLM Provider - Direct model access.
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
_model_cache = {}


class HuggingFaceTransformersConfig(LLMBaseConfig, name="huggingface_transformers"):
    """Configuration for HuggingFace Transformers LLM - loads model directly."""
    
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


class SimpleHFModel:
    """Simple wrapper that provides ainvoke interface for HF models."""
    
    def __init__(self, model_name: str, config: HuggingFaceTransformersConfig):
        self.model_name = model_name
        self.config = config
        
        # Get from cache
        if model_name not in _model_cache:
            raise ValueError(f"Model {model_name} not loaded in cache")
        
        cached = _model_cache[model_name]
        self.model = cached["model"]
        self.tokenizer = cached["tokenizer"]
        self.torch = cached["torch"]
    
    async def ainvoke(self, messages):
        """Generate response - matches LangChain interface."""
        # Convert messages to text
        if isinstance(messages, list) and len(messages) > 0:
            # Try using chat template
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback: just use the last message content
                last_msg = messages[-1]
                text = last_msg.get("content", str(last_msg)) if isinstance(last_msg, dict) else str(last_msg)
        else:
            text = str(messages)
        
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
        
        # Return object with content attribute (matches LangChain interface)
        class Response:
            def __init__(self, text):
                self.content = text
        
        return Response(content)


@register_llm_provider(config_type=HuggingFaceTransformersConfig)
async def huggingface_transformers_provider(config: HuggingFaceTransformersConfig, builder: Builder):
    """Load HuggingFace model and cache it."""
    logger.info(f"Loading HuggingFace model: {config.model_name}")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch required. Install: pip install transformers torch accelerate"
        )
    
    # Load model if not cached
    if config.model_name not in _model_cache:
        logger.info(f"Loading model from HuggingFace: {config.model_name}")
        
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
        
        logger.info(f"✓ Model loaded: {config.model_name} on device: {config.device}")
    else:
        logger.info(f"✓ Using cached model: {config.model_name}")
    
    # Create simple wrapper and store in provider info
    model_wrapper = SimpleHFModel(config.model_name, config)
    
    # Store wrapper in cache too for direct access
    _model_cache[f"{config.model_name}_wrapper"] = model_wrapper
    
    yield LLMProviderInfo(config=config, description=f"HuggingFace model: {config.model_name}")
    
    logger.info(f"Cleaning up HuggingFace model: {config.model_name}")


def get_huggingface_model(model_name: str):
    """Direct access to HF model wrapper - for use by middleware."""
    wrapper_key = f"{model_name}_wrapper"
    if wrapper_key in _model_cache:
        return _model_cache[wrapper_key]
    return None
