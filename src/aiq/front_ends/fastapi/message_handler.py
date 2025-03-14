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
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel
from pydantic import ValidationError
from starlette.endpoints import WebSocketEndpoint

from aiq.data_models.api_server import Error
from aiq.data_models.api_server import ErrorTypes
from aiq.data_models.api_server import SystemResponseContent
from aiq.data_models.api_server import TextContent
from aiq.data_models.api_server import WebSocketMessageStatus
from aiq.data_models.api_server import WebSocketMessageType
from aiq.data_models.api_server import WebSocketSystemInteractionMessage
from aiq.data_models.api_server import WebSocketSystemIntermediateStepMessage
from aiq.data_models.api_server import WebSocketSystemResponseTokenMessage
from aiq.data_models.api_server import WebSocketUserInteractionResponseMessage
from aiq.data_models.api_server import WebSocketUserMessage
from aiq.data_models.interactive import HumanPromptNotification
from aiq.data_models.interactive import HumanResponse
from aiq.data_models.interactive import HumanResponseNotification
from aiq.data_models.interactive import InteractionPrompt
from aiq.front_ends.fastapi.message_validator import MessageValidator

logger = logging.getLogger(__name__)


class MessageHandler:

    def __init__(self, websocket_reference: WebSocketEndpoint):
        self._websocket_reference: WebSocketEndpoint = websocket_reference
        self._message_validator: MessageValidator = MessageValidator()
        self._messages_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        self._out_going_messages_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._process_messages_task: asyncio.Task | None = None
        self._process_out_going_messages_task: asyncio.Task = None
        self._background_task: asyncio.Task = None
        self._message_parent_id: str = "default_id"
        self._workflow_schema_type: str = None
        self._user_interaction_response: asyncio.Future[TextContent] = asyncio.Future()

    @property
    def messages_queue(self) -> asyncio.Queue[dict[str, str]]:
        return self._messages_queue

    @property
    def background_task(self) -> asyncio.Task:
        return self._background_task

    @property
    def process_messages_task(self) -> asyncio.Task | None:
        return self._process_messages_task

    @process_messages_task.setter
    def process_messages_task(self, process_messages_task) -> None:
        self._process_messages_task = process_messages_task

    @property
    def process_out_going_messages_task(self) -> asyncio.Task:
        return self._process_out_going_messages_task

    @process_out_going_messages_task.setter
    def process_out_going_messages_task(self, process_out_going_messages_task) -> None:
        self._process_out_going_messages_task = process_out_going_messages_task

    async def process_messages(self) -> None:
        """
        Processes received messages from websocket and routes them appropriately.
        """
        while True:

            try:
                message: dict[str, Any] = await self._messages_queue.get()

                validated_message: BaseModel = await self._message_validator.validate_message(message)

                if (isinstance(validated_message, WebSocketUserMessage)):
                    await self.process_user_message(validated_message)

                if isinstance(
                        validated_message,
                    (  # noqa: E131
                        WebSocketSystemResponseTokenMessage,
                        WebSocketSystemIntermediateStepMessage,
                        WebSocketSystemInteractionMessage)):
                    await self._out_going_messages_queue.put(validated_message.model_dump())

                if (isinstance(validated_message, WebSocketUserInteractionResponseMessage)):
                    user_content = await self.process_user_message_content(validated_message)
                    self._user_interaction_response.set_result(user_content)
            except (asyncio.CancelledError):
                break

        return None

    async def process_user_message_content(
            self, user_content: WebSocketUserMessage | WebSocketUserInteractionResponseMessage) -> BaseModel | None:
        """
        Processes the contents of a user message.

        :param user_content: Incoming content data model.
        :return: A validated Pydantic user content model or None if not found.
        """

        for user_message in user_content.content.messages[::-1]:
            if (user_message.role == "user"):

                for attachment in user_message.content:

                    if isinstance(attachment, TextContent):
                        return attachment

        return None

    async def process_user_message(self, message_as_validated_type: WebSocketUserMessage) -> None:
        """
        Process user messages and routes them appropriately.

        :param message_as_validated_type: A WebSocketUserMessage Data Model instance.
        """

        try:
            self._message_parent_id = message_as_validated_type.id
            self._workflow_schema_type = message_as_validated_type.schema_type

            content: BaseModel | None = await self.process_user_message_content(message_as_validated_type)

            if content is None:
                raise ValueError(f"User message content could not be found: {message_as_validated_type}")

            if isinstance(content, TextContent) and (self._background_task is None):

                await self._process_response()
                self._background_task = asyncio.create_task(
                    self._websocket_reference.workflow_schema_type.get(self._workflow_schema_type)(
                        content.text)).add_done_callback(
                            lambda task: asyncio.create_task(self._on_process_stream_task_done(task)))

        except ValueError as e:
            logger.error("User message content not found: %s", str(e), exc_info=True)
            await self.create_websocket_message(data_model=Error(code=ErrorTypes.INVALID_USER_MESSAGE_CONTENT,
                                                                 message="User message content could not be found",
                                                                 details=str(e)),
                                                message_type=WebSocketMessageType.ERROR_MESSAGE,
                                                status=WebSocketMessageStatus.IN_PROGRESS)

    async def create_websocket_message(self,
                                       data_model: BaseModel,
                                       message_type: str | None = None,
                                       status: str = WebSocketMessageStatus.IN_PROGRESS) -> None:
        """
        Creates a websocket message that will be ready for routing based on message type or data model.

        :param data_model: Message content model.
        :param message_type: Message content model.
        :param status: Message content model.
        """
        try:
            message: BaseModel | None = None

            if message_type is None:
                message_type = await self._message_validator.resolve_message_type_by_data(data_model)

            message_schema: type[BaseModel] = await self._message_validator.get_message_schema_by_type(message_type)

            if 'id' in data_model.model_fields:
                message_id: str = data_model.id
            else:
                message_id = str(uuid.uuid4())

            content: BaseModel = await self._message_validator.convert_data_to_message_content(data_model)

            if issubclass(message_schema, WebSocketSystemResponseTokenMessage):
                message = await self._message_validator.create_system_response_token_message(
                    message_id=message_id, parent_id=self._message_parent_id, content=content, status=status)

            elif issubclass(message_schema, WebSocketSystemIntermediateStepMessage):
                message = await self._message_validator.create_system_intermediate_step_message(
                    message_id=message_id,
                    parent_id=await self._message_validator.get_intermediate_step_parent_id(data_model),
                    content=content,
                    status=status)

            elif issubclass(message_schema, WebSocketSystemInteractionMessage):
                message = await self._message_validator.create_system_interaction_message(
                    message_id=message_id, parent_id=self._message_parent_id, content=content, status=status)

            elif isinstance(content, Error):
                raise ValidationError(f"Invalid input data creating websocket message. {data_model.model_dump_json()}")

            elif issubclass(message_schema, Error):
                raise TypeError(f"Invalid message type: {message_type}")

            elif (message is None):
                raise ValueError(
                    f"Message type could not be resolved by input data model: {data_model.model_dump_json()}")

        except (ValidationError, TypeError, ValueError) as e:
            logger.error("A data vaidation error ocurred creating websocket message: %s", str(e), exc_info=True)
            message = await self._message_validator.create_system_response_token_message(
                message_type=WebSocketMessageType.ERROR_MESSAGE,
                content=Error(code=ErrorTypes.UNKNOWN_ERROR, message="default", details=str(e)))

        finally:
            await self._messages_queue.put(message.model_dump())

    async def _on_process_stream_task_done(self, task: asyncio.Task) -> None:
        await self.create_websocket_message(data_model=SystemResponseContent(),
                                            message_type=WebSocketMessageType.RESPONSE_MESSAGE,
                                            status=WebSocketMessageStatus.COMPLETE)

        return None

    async def process_out_going_messages(self, websocket: WebSocket) -> None:
        """
        Spawns out going message processing task.

        :param websocket: Websocket instance.
        """
        while True:
            try:
                out_going_message = await self._out_going_messages_queue.get()
                await self._websocket_reference.on_send(websocket, out_going_message)

            except (asyncio.CancelledError, ValidationError):
                break

        return None

    async def _process_response(self):
        self._websocket_reference.process_response_event.set()

    async def _pause_response(self):
        self._websocket_reference.process_response_event.clear()

    async def __reset_user_interaction_response(self):
        self._user_interaction_response = asyncio.Future()

    async def human_interaction(self, prompt: InteractionPrompt) -> HumanResponse:
        """
        Registered human interaction callback that processes human interactions and returns
        responses from websocket connection.

        :param prompt: Incoming interaction content data model.
        :return: A Text Content Base Pydantic model.
        """
        await self.create_websocket_message(data_model=prompt.content,
                                            message_type=WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE,
                                            status=WebSocketMessageStatus.IN_PROGRESS)

        if (isinstance(prompt.content, HumanPromptNotification)):
            return HumanResponseNotification()

        user_message_repsonse_content: TextContent = await self._user_interaction_response
        interaction_response: HumanResponse = await self._message_validator.convert_text_content_to_human_response(
            user_message_repsonse_content, prompt.content)

        await self.__reset_user_interaction_response()
        await self._process_response()

        return interaction_response
