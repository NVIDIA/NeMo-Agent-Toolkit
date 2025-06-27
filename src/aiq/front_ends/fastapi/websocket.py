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
import json
import logging
from typing import Any, Dict

from fastapi import WebSocket

from aiq.data_models.api_server import (
    MessageRequest,
    ModeGetResponse,
    ModeSetRequest,
)
from aiq.front_ends.fastapi.message_handler import MessageHandler
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    def __init__(self, session_manager: AIQSessionManager):
        self._session_manager = session_manager
        self._message_handler = MessageHandler(session_manager)

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                if message_data.get("type") == "message":
                    await self._handle_message(websocket, message_data)
                elif message_data.get("type") == "get_mode":
                    await self._handle_get_mode(websocket)
                elif message_data.get("type") == "set_mode":
                    await self._handle_set_mode(websocket, message_data)
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_data.get('type')}"
                    }))
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()

    async def _handle_message(self, websocket: WebSocket, message_data: Dict[str, Any]):
        """Handle a chat message."""
        try:
            request = MessageRequest(
                message=message_data["message"],
                conversation_id=message_data.get("conversation_id")
            )
            
            response = await self._message_handler.send_message(request)
            
            await websocket.send_text(json.dumps({
                "type": "message_response",
                "conversation_id": response.conversation_id,
                "response": response.response,
                "mode": response.mode
            }))
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def _handle_get_mode(self, websocket: WebSocket):
        """Handle get mode request."""
        try:
            response = await self._message_handler.get_mode()
            
            await websocket.send_text(json.dumps({
                "type": "mode_response",
                "current_mode": response.current_mode,
                "available_modes": response.available_modes
            }))
            
        except Exception as e:
            logger.error(f"Error getting mode: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def _handle_set_mode(self, websocket: WebSocket, message_data: Dict[str, Any]):
        """Handle set mode request."""
        try:
            request = ModeSetRequest(mode=message_data["mode"])
            response = await self._message_handler.set_mode(request)
            
            await websocket.send_text(json.dumps({
                "type": "mode_set_response",
                "success": response.success,
                "message": response.message,
                "current_mode": response.current_mode
            }))
            
        except Exception as e:
            logger.error(f"Error setting mode: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))

