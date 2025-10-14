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

import logging
import re
from typing import Any

from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from nat.builder.context import Context
from nat.data_models.api_server import ChatRequest, Message, UserMessageContentRoleType
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem
from .state import AutoMemoryWrapperState

logger = logging.getLogger(__name__)


class AutoMemoryWrapperGraph:
    """
    Wraps any NAT agent to add automatic memory capture and retrieval.

    The wrapper treats the inner agent as a black-box function that receives
    a ChatRequest with multiple messages (including system messages with memory
    context). The inner agent manages its own internal state (ReActGraphState,
    ReWOOGraphState, etc.) - the wrapper never sees it.
    """

    def __init__(
        self,
        inner_agent_fn,  # Inner agent as a Function (receives ChatRequest with multiple messages)
        memory_editor: MemoryEditor,  # Zep/Mem0/Redis memory client
        save_user_messages: bool = True,  # Auto-save user messages
        retrieve_memory: bool = True,  # Auto-retrieve before agent
        save_ai_responses: bool = True,  # Auto-save agent responses
        search_params: dict[str, Any] | None = None,  # Backend-specific search parameters
        add_params: dict[str, Any] | None = None  # Backend-specific add parameters
    ):
        self.inner_agent_fn = inner_agent_fn
        self.memory_editor = memory_editor
        self.save_user_messages = save_user_messages
        self.retrieve_memory = retrieve_memory
        self.save_ai_responses = save_ai_responses
        self.search_params = search_params or {}
        self.add_params = add_params or {}

    @staticmethod
    def _langchain_message_to_nat_message(lc_message: BaseMessage) -> Message:
        """
        Convert LangChain message to NAT Message format.

        This is necessary to construct a proper ChatRequest with all messages
        (including system messages with memory context) to pass to the inner agent.
        """
        if isinstance(lc_message, HumanMessage):
            role = UserMessageContentRoleType.USER
        elif isinstance(lc_message, AIMessage):
            role = UserMessageContentRoleType.ASSISTANT
        elif isinstance(lc_message, SystemMessage):
            role = UserMessageContentRoleType.SYSTEM
        else:
            # Default to user for unknown message types
            role = UserMessageContentRoleType.USER

        return Message(role=role, content=str(lc_message.content))

    async def capture_user_message_node(self, state: AutoMemoryWrapperState):
        """Captures user message to memory thread"""
        if not self.save_user_messages or not state.messages:
            return state

        # Get the latest user message
        user_message = state.messages[-1]
        if isinstance(user_message, HumanMessage):
            # Get user_id from Context (thread_id is retrieved within the memory editor)
            user_id = Context.get().user_id or "default_NAT_user"

            # Add to memory, passing user_id as positional argument
            await self.memory_editor.add_items([
                MemoryItem(
                    conversation=[{"role": "user", "content": str(user_message.content)}]
                )
            ], user_id, **self.add_params)
        return state

    async def memory_retrieve_node(self, state: AutoMemoryWrapperState):
        """Retrieves relevant memory from memory store"""
        if not self.retrieve_memory or not state.messages:
            return state

        # Get the latest user message
        user_message = state.messages[-1]

        # Get user_id from Context (thread_id is retrieved within the memory editor)
        user_id = Context.get().user_id or "default_NAT_user"

        # Retrieve formatted memory from memory provider
        memory_string = await self.memory_editor.retrieve_memory(
            query=user_message.content,  # Reasonable default for memory retrieval
            user_id=user_id,
            **self.search_params  # User-configured params (e.g., top_k, mode)
        )

        # Inject memory as system message if available
        if memory_string:
            memory_message = SystemMessage(
                content=f"Relevant context from memory:\n{memory_string}"
            )
            # Insert before the last user message
            state.messages.insert(-1, memory_message)

        return state

    async def inner_agent_node(self, state: AutoMemoryWrapperState):
        """
        Calls the inner agent with a ChatRequest containing all messages.

        The inner agent receives a ChatRequest with multiple messages (including
        system messages with memory context), processes them using its own internal
        state (ReActGraphState, ReWOOGraphState, etc.), and returns a ChatResponse.
        """
        # Convert all LangChain messages to NAT Message format
        nat_messages = [self._langchain_message_to_nat_message(msg) for msg in state.messages]
        chat_request = ChatRequest(messages=nat_messages)

        # Call inner agent with ChatRequest - it manages its own state internally
        response = await self.inner_agent_fn.ainvoke(chat_request)

        # Extract content from response based on type
        if hasattr(response, 'choices') and response.choices:
            # ChatResponse object - extract from choices[0].message.content
            response_text = response.choices[0].message.content or ""
        elif hasattr(response, 'output'):
            # GenerateResponse object - use output field
            response_text = response.output
        elif hasattr(response, 'value'):
            # Some other response type with value field
            response_text = str(response.value)
        elif isinstance(response, str):
            # Already a string
            response_text = response
        else:
            # Last resort: convert to string
            response_text = str(response)

        # Add response to wrapper state
        state.messages.append(AIMessage(content=response_text))
        return state

    async def capture_ai_response_node(self, state: AutoMemoryWrapperState):
        """Captures agent response to memory"""
        if not self.save_ai_responses or not state.messages:
            return state

        # Get the latest AI message
        ai_message = state.messages[-1]
        if isinstance(ai_message, AIMessage):
            # Get user_id from Context (thread_id is retrieved within the memory editor)
            user_id = Context.get().user_id or "default_NAT_user"

            # Add to memory, passing user_id as positional argument
            await self.memory_editor.add_items([
                MemoryItem(
                    conversation=[{"role": "assistant", "content": str(ai_message.content)}]
                )
            ], user_id, **self.add_params)
        return state

    def build_graph(self):
        """Wraps inner agent with memory nodes"""
        workflow = StateGraph(AutoMemoryWrapperState)

        # Add nodes
        if self.save_user_messages:
            workflow.add_node("capture_user_message", self.capture_user_message_node)
        if self.retrieve_memory:
            workflow.add_node("memory_retrieve", self.memory_retrieve_node)
        workflow.add_node("inner_agent", self.inner_agent_node)
        if self.save_ai_responses:
            workflow.add_node("capture_ai_response", self.capture_ai_response_node)

        # Connect nodes based on enabled features
        workflow.set_entry_point("capture_user_message" if self.save_user_messages else
                                 "memory_retrieve" if self.retrieve_memory else
                                 "inner_agent")

        if self.save_user_messages and self.retrieve_memory:
            workflow.add_edge("capture_user_message", "memory_retrieve")
            workflow.add_edge("memory_retrieve", "inner_agent")
        elif self.save_user_messages:
            workflow.add_edge("capture_user_message", "inner_agent")
        elif self.retrieve_memory:
            workflow.add_edge("memory_retrieve", "inner_agent")

        if self.save_ai_responses:
            workflow.add_edge("inner_agent", "capture_ai_response")
            workflow.set_finish_point("capture_ai_response")
        else:
            workflow.set_finish_point("inner_agent")

        return workflow.compile()
