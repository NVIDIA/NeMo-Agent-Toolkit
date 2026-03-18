# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Simple Completion Function for NAT

This module provides a simple completion function that can handle
natural language queries and perform basic text completion tasks.
Supports OpenAI-style message history when used with the chat completions API.
"""

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import Usage
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.type_converter import GlobalTypeConverter


class ChatCompletionConfig(FunctionBaseConfig, name="chat_completion"):
    """Configuration for the Chat Completion Function."""

    system_prompt: str = Field(("You are a helpful AI assistant. Provide clear, accurate, and helpful "
                                "responses to user queries. You can give general advice, recommendations, "
                                "tips, and engage in conversation. Be helpful and informative."),
                               description="The system prompt to use for chat completion.")

    llm_name: LLMRef = Field(description="The LLM to use for generating responses.")


def _messages_to_langchain_messages(
    nat_messages: list,
    system_prompt: str,
):
    """Convert NAT Message list to LangChain BaseMessage list with system prompt prepended if needed."""
    from langchain_core.messages.utils import convert_to_messages

    message_dicts = [m.model_dump() for m in nat_messages]
    has_system = any(d.get("role") == "system" for d in message_dicts)
    if not has_system and system_prompt:
        message_dicts = [{"role": "system", "content": system_prompt}] + message_dicts
    return convert_to_messages(message_dicts)


@register_function(config_type=ChatCompletionConfig)
async def register_chat_completion(config: ChatCompletionConfig, builder: Builder):
    """Registers a chat completion function that can handle natural language queries and full message history."""

    # Get the LLM from the builder context using the configured LLM reference
    # Use LangChain/LangGraph framework wrapper since we're using LangChain/LangGraph-based LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _chat_completion(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        """Chat completion that supports OpenAI-style message history.

        Accepts either a single input_message (string) or a full conversation
        (messages array). When messages are provided, the full history is sent
        to the LLM for context-aware responses.

        Args:
            chat_request_or_message: Either a string input or OpenAI-style messages array.

        Returns:
            ChatResponse when input is a conversation; str when input is a single message.
        """
        try:
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)

            # Build LangChain message list from full conversation (OpenAI message history)
            lc_messages = _messages_to_langchain_messages(
                message.messages,
                config.system_prompt,
            )

            # Generate response using the LLM with full message history
            response = await llm.ainvoke(lc_messages)

            if isinstance(response, str):
                output_text = response
            else:
                output_text = response.text() if hasattr(response, "text") else str(response.content)

            # Approximate usage for API compatibility
            prompt_tokens = sum(len(str(m.content).split()) for m in message.messages)
            completion_tokens = len(output_text.split()) if output_text else 0
            total_tokens = prompt_tokens + completion_tokens
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            chat_response = ChatResponse.from_string(output_text, usage=usage)

            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(chat_response, to_type=str)
            return chat_response

        except Exception as e:
            last_content = ""
            try:
                msg = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
                if msg.messages:
                    last = msg.messages[-1].content
                    last_content = last if isinstance(last, str) else str(last)
            except Exception:
                pass
            return (f"I apologize, but I encountered an error while processing your "
                    f"query: '{last_content}'. Please try rephrasing your question or try "
                    f"again later. Error: {str(e)}")

    yield FunctionInfo.from_fn(
        _chat_completion,
        description=getattr(config, "description", "Chat completion"),
    )
