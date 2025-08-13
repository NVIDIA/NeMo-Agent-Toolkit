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
Simple Completion Function for AIQ Toolkit

This module provides a simple completion function that can handle
natural language queries and perform basic text completion tasks.
"""

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class ChatCompletionConfig(FunctionBaseConfig, name="chat_completion"):
    """Configuration for the Chat Completion Function."""

    system_prompt: str = Field(("You are a helpful AI assistant. Provide clear, accurate, and helpful "
                                "responses to user queries. You can give general advice, recommendations, "
                                "tips, and engage in conversation. Be helpful and informative."),
                               description="The system prompt to use for chat completion.")

    llm_name: LLMRef = Field(description="The LLM to use for generating responses.")


@register_function(config_type=ChatCompletionConfig)
async def register_chat_completion(config: ChatCompletionConfig, builder: Builder):
    """Registers a chat completion function that can handle natural language queries."""

    # Get the LLM from the builder context using the configured LLM reference
    # Use LangChain framework wrapper since we're using LangChain-based LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _chat_completion(query: str) -> str:
        """A simple chat completion function that responds to natural language queries.

        Args:
            query: The user's natural language query

        Returns:
            A helpful response to the query
        """
        try:
            # Create a simple prompt with the system message and user query
            prompt = f"{config.system_prompt}\n\nUser: {query}\n\nAssistant:"

            # Generate response using the LLM
            response = await llm.ainvoke(prompt)

            return response

        except Exception as e:
            # Fallback response if LLM call fails
            return (f"I apologize, but I encountered an error while processing your "
                    f"query: '{query}'. Please try rephrasing your question or try "
                    f"again later. Error: {str(e)}")

    yield _chat_completion
