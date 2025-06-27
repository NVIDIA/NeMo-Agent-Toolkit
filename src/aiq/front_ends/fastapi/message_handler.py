# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from aiq.data_models.api_server import (
    ConversationDeleteRequest,
    ConversationDeleteResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageRequest,
    MessageResponse,
    ModeGetResponse,
    ModeSetRequest,
    ModeSetResponse,
)
from aiq.data_models.config import AIQConfig
from aiq.data_models.conversation import Conversation, ConversationMessage
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


class MessageHandler:
    def __init__(self, session_manager: AIQSessionManager):
        self._session_manager = session_manager
        self._conversations: Dict[str, Conversation] = {}
        self._current_mode = "DEFAULT"  # Single mode for single server

    async def send_message(self, request: MessageRequest) -> MessageResponse:
        """Send a message and get a response from the workflow."""
        try:
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Get or create conversation
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                conversation = Conversation(
                    conversation_id=conversation_id,
                    messages=[],
                    mode=self._current_mode
                )
                self._conversations[conversation_id] = conversation

            # Add user message to conversation
            user_message = ConversationMessage(
                role="user",
                content=request.message,
                timestamp=request.timestamp
            )
            conversation.messages.append(user_message)

            # Get workflow response using session manager
            async with self._session_manager.session() as session:
                async with session.run(request.message) as runner:
                    workflow_response = await runner
            
            # Add assistant message to conversation
            assistant_message = ConversationMessage(
                role="assistant", 
                content=workflow_response,
                timestamp=None  # Will be set automatically
            )
            conversation.messages.append(assistant_message)

            return MessageResponse(
                conversation_id=conversation_id,
                response=workflow_response,
                mode=self._current_mode
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise

    async def get_conversations(self) -> ConversationListResponse:
        """Get all conversations."""
        conversations = [
            ConversationResponse(
                conversation_id=conv.conversation_id,
                messages=conv.messages,
                mode=conv.mode
            )
            for conv in self._conversations.values()
        ]
        
        return ConversationListResponse(conversations=conversations)

    async def delete_conversation(self, request: ConversationDeleteRequest) -> ConversationDeleteResponse:
        """Delete a conversation."""
        if request.conversation_id in self._conversations:
            del self._conversations[request.conversation_id]
            return ConversationDeleteResponse(success=True, message="Conversation deleted successfully")
        else:
            return ConversationDeleteResponse(success=False, message="Conversation not found")

    async def get_mode(self) -> ModeGetResponse:
        """Get the current mode - always DEFAULT for single server."""
        return ModeGetResponse(current_mode=self._current_mode, available_modes=["DEFAULT"])

    async def set_mode(self, request: ModeSetRequest) -> ModeSetResponse:
        """Set mode - not supported in single server mode."""
        return ModeSetResponse(
            success=False, 
            message="Mode switching not supported in single server mode. Use separate servers for different configurations.",
            current_mode=self._current_mode
        )
