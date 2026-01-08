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
"""LangChain-compatible wrapper for local HuggingFace Transformers models."""

import asyncio
import logging
import threading
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.outputs import ChatResult
from pydantic import ConfigDict

from nat.llm.huggingface_llm import HuggingFaceConfig

logger = logging.getLogger(__name__)


class ChatHuggingFace(BaseChatModel):
    """LangChain-compatible wrapper for local HuggingFace models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _model_name: str
    _hf_config: HuggingFaceConfig
    _model: Any
    _tokenizer: Any
    _torch: Any

    def __init__(self, model_name: str, hf_config: HuggingFaceConfig, cached: dict[str, Any]):
        super().__init__()
        self._model_name = model_name
        self._hf_config = hf_config
        self._model = cached["model"]
        self._tokenizer = cached["tokenizer"]
        self._torch = cached["torch"]

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _prepare_text(self, messages: list[BaseMessage] | list[dict] | str) -> str:
        """Convert messages to text using chat template or fallback."""
        if isinstance(messages, list) and len(messages) > 0:
            if hasattr(messages[0], "type") and hasattr(messages[0], "content"):
                messages = [{
                    "role": msg.type, "content": msg.content
                } for msg in messages]  # type: ignore[attr-defined]

            try:
                text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
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
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._prepare_text(messages)
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=self._hf_config.max_new_tokens,
                temperature=self._hf_config.temperature if self._hf_config.temperature > 0 else None,
                do_sample=self._hf_config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)

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
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            logger.debug("TextIteratorStreamer not available, falling back to non-streaming")
            result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            full_message = result.generations[0].message
            chunk = AIMessageChunk(content=full_message.content)
            yield ChatGenerationChunk(message=chunk)
            return

        text = self._prepare_text(messages)
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True, skip_prompt=True)

        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": self._hf_config.max_new_tokens,
            "temperature": self._hf_config.temperature if self._hf_config.temperature > 0 else None,
            "do_sample": self._hf_config.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        try:
            for token_text in streamer:
                await asyncio.sleep(0)
                chunk = AIMessageChunk(content=token_text)
                yield ChatGenerationChunk(message=chunk)
        finally:
            thread.join()

    def bind_tools(self, tools, **kwargs):
        """Bind tools to the LLM. Returns self for compatibility."""
        return self

    def bind(self, **kwargs):
        """Bind additional parameters to the LLM. Returns self for compatibility."""
        return self
