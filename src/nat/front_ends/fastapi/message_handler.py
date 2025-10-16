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
import typing
import uuid
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from nat.authentication.interfaces import FlowHandlerBase
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.api_server import SystemResponseContent
from nat.data_models.api_server import TextContent
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.api_server import UserMessages
from nat.data_models.api_server import WebSocketMessageStatus
from nat.data_models.api_server import WebSocketMessageType
from nat.data_models.api_server import WebSocketSystemInteractionMessage
from nat.data_models.api_server import WebSocketSystemIntermediateStepMessage
from nat.data_models.api_server import WebSocketSystemResponseTokenMessage
from nat.data_models.api_server import WebSocketUserInteractionResponseMessage
from nat.data_models.api_server import WebSocketUserMessage
from nat.data_models.api_server import WorkflowSchemaType
from nat.data_models.interactive import HumanPromptNotification
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import HumanResponseNotification
from nat.data_models.interactive import InteractionPrompt
from nat.front_ends.fastapi.message_validator import MessageValidator
from nat.front_ends.fastapi.response_helpers import generate_streaming_response
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class WebSocketMessageHandler:

    def __init__(self, socket: WebSocket, session_manager: SessionManager, step_adaptor: StepAdaptor):
        self._socket: WebSocket = socket
        self._session_manager: SessionManager = session_manager
        self._step_adaptor: StepAdaptor = step_adaptor

        self._message_validator: MessageValidator = MessageValidator()
        self._running_workflow_task: asyncio.Task | None = None
        self._message_parent_id: str = "default_id"
        self._conversation_id: str | None = None
        self._workflow_schema_type: str | None = None
        self._user_interaction_response: asyncio.Future[TextContent] | None = None

        self._flow_handler: FlowHandlerBase | None = None

        self._schema_output_mapping: dict[str, type[BaseModel] | type[None]] = {
            WorkflowSchemaType.GENERATE: self._session_manager.workflow.single_output_schema,
            WorkflowSchemaType.CHAT: ChatResponse,
            WorkflowSchemaType.CHAT_STREAM: ChatResponseChunk,
            WorkflowSchemaType.GENERATE_STREAM: self._session_manager.workflow.streaming_output_schema,
        }

    def set_flow_handler(self, flow_handler: FlowHandlerBase) -> None:
        self._flow_handler = flow_handler

    async def __aenter__(self) -> "WebSocketMessageHandler":
        await self._socket.accept()

        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:

        # TODO: Handle the exit
        pass

    async def run(self) -> None:
        """
        Processes received messages from websocket and routes them appropriately.
        """
        while True:

            try:

                message: dict[str, Any] = await self._socket.receive_json()

                validated_message: BaseModel = await self._message_validator.validate_message(message)

                # Received a request to start a workflow
                if (isinstance(validated_message, WebSocketUserMessage)):
                    await self.process_workflow_request(validated_message)

                elif isinstance(
                        validated_message,
                        WebSocketSystemResponseTokenMessage | WebSocketSystemIntermediateStepMessage
                        | WebSocketSystemInteractionMessage):
                    # These messages are already handled by self.create_websocket_message(data_model=value, …)
                    # No further processing is needed here.
                    pass

                elif (isinstance(validated_message, WebSocketUserInteractionResponseMessage)):
                    user_content = await self._process_websocket_user_interaction_response_message(validated_message)
                    assert self._user_interaction_response is not None
                    self._user_interaction_response.set_result(user_content)
            except (asyncio.CancelledError, WebSocketDisconnect):
                # TODO: Handle the disconnect
                break

    def _extract_last_user_message_content(self, messages: list[UserMessages]) -> TextContent:
        """
        Extracts the last user's TextContent from a list of messages.

        Args:
            messages: List of UserMessages.

        Returns:
            TextContent object from the last user message.

        Raises:
            ValueError: If no user text content is found.
        """
        for user_message in messages[::-1]:
            if user_message.role == UserMessageContentRoleType.USER:
                for attachment in user_message.content:
                    if isinstance(attachment, TextContent):
                        return attachment
        raise ValueError("No user text content found in messages.")

    async def _process_websocket_user_interaction_response_message(
            self, user_content: WebSocketUserInteractionResponseMessage) -> TextContent:
        """
        Processes a WebSocketUserInteractionResponseMessage.
        """
        return self._extract_last_user_message_content(user_content.content.messages)

    async def _process_websocket_user_message(self, user_content: WebSocketUserMessage) -> ChatRequest | str:
        """
        Processes a WebSocketUserMessage based on schema type.
        """
        if self._workflow_schema_type in [WorkflowSchemaType.CHAT, WorkflowSchemaType.CHAT_STREAM]:
            return ChatRequest(**user_content.content.model_dump(include={"messages"}))

        elif self._workflow_schema_type in [WorkflowSchemaType.GENERATE, WorkflowSchemaType.GENERATE_STREAM]:
            return self._extract_last_user_message_content(user_content.content.messages).text

        raise ValueError("Unsupported workflow schema type for WebSocketUserMessage")

    async def process_workflow_request(self, user_message_as_validated_type: WebSocketUserMessage) -> None:
        """
        Process user messages and routes them appropriately.

        Args:
            user_message_as_validated_type (WebSocketUserMessage): The validated user message to process.
        """

        try:
            self._message_parent_id = user_message_as_validated_type.id
            self._workflow_schema_type = user_message_as_validated_type.schema_type
            self._conversation_id = user_message_as_validated_type.conversation_id

            message_content: typing.Any = await self._process_websocket_user_message(user_message_as_validated_type)

            if (self._running_workflow_task is None):

                def _done_callback(_task: asyncio.Task):
                    self._running_workflow_task = None

                self._running_workflow_task = asyncio.create_task(
                    self._run_workflow(payload=message_content,
                                       user_message_id=self._message_parent_id,
                                       conversation_id=self._conversation_id,
                                       result_type=self._schema_output_mapping[self._workflow_schema_type],
                                       output_type=self._schema_output_mapping[
                                           self._workflow_schema_type])).add_done_callback(_done_callback)

        except ValueError as e:
            logger.exception("User message content not found: %s", str(e))
            await self.create_websocket_message(data_model=Error(code=ErrorTypes.INVALID_USER_MESSAGE_CONTENT,
                                                                 message="User message content could not be found",
                                                                 details=str(e)),
                                                message_type=WebSocketMessageType.ERROR_MESSAGE,
                                                status=WebSocketMessageStatus.IN_PROGRESS)

    async def create_websocket_message(self,
                                       data_model: BaseModel,
                                       message_type: str | None = None,
                                       status: WebSocketMessageStatus = WebSocketMessageStatus.IN_PROGRESS) -> None:
        """
        Creates a websocket message that will be ready for routing based on message type or data model.

        Args:
            data_model (BaseModel): Message content model.
            message_type (str | None): Message content model.
            status (WebSocketMessageStatus): Message content model.
        """
        try:
            message: BaseModel | None = None

            if message_type is None:
                message_type = await self._message_validator.resolve_message_type_by_data(data_model)

            message_schema: type[BaseModel] = await self._message_validator.get_message_schema_by_type(message_type)

            if hasattr(data_model, 'id'):
                message_id: str = str(getattr(data_model, 'id'))
            else:
                message_id = str(uuid.uuid4())

            content: BaseModel = await self._message_validator.convert_data_to_message_content(data_model)

            if issubclass(message_schema, WebSocketSystemResponseTokenMessage):
                message = await self._message_validator.create_system_response_token_message(
                    message_id=message_id,
                    parent_id=self._message_parent_id,
                    conversation_id=self._conversation_id,
                    content=content,
                    status=status)

            elif issubclass(message_schema, WebSocketSystemIntermediateStepMessage):
                message = await self._message_validator.create_system_intermediate_step_message(
                    message_id=message_id,
                    parent_id=await self._message_validator.get_intermediate_step_parent_id(data_model),
                    conversation_id=self._conversation_id,
                    content=content,
                    status=status)

            elif issubclass(message_schema, WebSocketSystemInteractionMessage):
                message = await self._message_validator.create_system_interaction_message(
                    message_id=message_id,
                    parent_id=self._message_parent_id,
                    conversation_id=self._conversation_id,
                    content=content,
                    status=status)

            elif isinstance(content, Error):
                raise ValidationError(f"Invalid input data creating websocket message. {data_model.model_dump_json()}")

            elif issubclass(message_schema, Error):
                raise TypeError(f"Invalid message type: {message_type}")

            elif (message is None):
                raise ValueError(
                    f"Message type could not be resolved by input data model: {data_model.model_dump_json()}")

        except (ValidationError, TypeError, ValueError) as e:
            logger.exception("A data vaidation error ocurred creating websocket message: %s", str(e))
            message = await self._message_validator.create_system_response_token_message(
                message_type=WebSocketMessageType.ERROR_MESSAGE,
                conversation_id=self._conversation_id,
                content=Error(code=ErrorTypes.UNKNOWN_ERROR, message="default", details=str(e)))

        finally:
            if (message is not None):
                await self._socket.send_json(message.model_dump())

    async def human_interaction_callback(self, prompt: InteractionPrompt) -> HumanResponse:
        """
        Registered human interaction callback that processes human interactions and returns
        responses from websocket connection.

        Args:
            prompt: Incoming interaction content data model.

        Returns:
            A Text Content Base Pydantic model.
        """

        # First create a future from the loop for the human response
        human_response_future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()

        # Then add the future to the outstanding human prompts dictionary
        self._user_interaction_response = human_response_future

        try:

            await self.create_websocket_message(data_model=prompt.content,
                                                message_type=WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE,
                                                status=WebSocketMessageStatus.IN_PROGRESS)

            if (isinstance(prompt.content, HumanPromptNotification)):

                return HumanResponseNotification()

            # Wait for the human response future to complete
            text_content: TextContent = await human_response_future

            interaction_response: HumanResponse = await self._message_validator.convert_text_content_to_human_response(
                text_content, prompt.content)

            return interaction_response

        finally:
            # Delete the future from the outstanding human prompts dictionary
            self._user_interaction_response = None

    async def _run_workflow(self,
                            payload: typing.Any,
                            user_message_id: str | None = None,
                            conversation_id: str | None = None,
                            result_type: type | None = None,
                            output_type: type | None = None) -> None:

        try:
            auth_callback = self._flow_handler.authenticate if self._flow_handler else None
            async with self._session_manager.session(user_message_id=user_message_id,
                                                     conversation_id=conversation_id,
                                                     http_connection=self._socket,
                                                     user_input_callback=self.human_interaction_callback,
                                                     user_authentication_callback=auth_callback) as session:

                async for value in generate_streaming_response(payload,
                                                               session_manager=session,
                                                               streaming=True,
                                                               step_adaptor=self._step_adaptor,
                                                               result_type=result_type,
                                                               output_type=output_type):

                    if not isinstance(value, ResponseSerializable):
                        value = ResponsePayloadOutput(payload=value)

                    await self.create_websocket_message(data_model=value, status=WebSocketMessageStatus.IN_PROGRESS)

        finally:
            await self.create_websocket_message(data_model=SystemResponseContent(),
                                                message_type=WebSocketMessageType.RESPONSE_MESSAGE,
                                                status=WebSocketMessageStatus.COMPLETE)
