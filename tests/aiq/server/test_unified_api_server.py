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

import datetime
import json
import re

import httpx
import pytest
import pytest_asyncio
import yaml
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from pydantic import BaseModel
from pydantic import ValidationError

from aiq.builder.context import AIQContext
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQChoice
from aiq.data_models.api_server import AIQChoiceMessage
from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.api_server import AIQResponsePayloadOutput
from aiq.data_models.api_server import Error
from aiq.data_models.api_server import ErrorTypes
from aiq.data_models.api_server import SystemIntermediateStepContent
from aiq.data_models.api_server import SystemResponseContent
from aiq.data_models.api_server import TextContent
from aiq.data_models.api_server import WebSocketMessageType
from aiq.data_models.api_server import WebSocketSystemInteractionMessage
from aiq.data_models.api_server import WebSocketSystemIntermediateStepMessage
from aiq.data_models.api_server import WebSocketSystemResponseTokenMessage
from aiq.data_models.api_server import WebSocketUserInteractionResponseMessage
from aiq.data_models.api_server import WebSocketUserMessage
from aiq.data_models.interactive import BinaryHumanPromptOption
from aiq.data_models.interactive import HumanPromptBinary
from aiq.data_models.interactive import HumanPromptCheckbox
from aiq.data_models.interactive import HumanPromptDropdown
from aiq.data_models.interactive import HumanPromptNotification
from aiq.data_models.interactive import HumanPromptRadio
from aiq.data_models.interactive import HumanPromptText
from aiq.data_models.interactive import HumanResponseBinary
from aiq.data_models.interactive import HumanResponseCheckbox
from aiq.data_models.interactive import HumanResponseDropdown
from aiq.data_models.interactive import HumanResponseRadio
from aiq.data_models.interactive import HumanResponseText
from aiq.data_models.interactive import MultipleChoiceOption
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from aiq.front_ends.fastapi.message_validator import MessageValidator
from aiq.runtime.loader import load_config


class AppConfig(BaseModel):
    host: str
    ws: str
    port: int
    config_filepath: str
    input: str


class EndpointConfig(BaseModel):
    generate: str
    chat: str
    generate_stream: str
    chat_stream: str


class Config(BaseModel):
    app: AppConfig
    endpoint: EndpointConfig


class TEST(BaseModel):
    test: str = "TEST"


# ======== Raw WebSocket Message Schemas ========
user_message = {
    "type": "user_message",
    "schema_type": "chat",
    "id": "string",
    "conversation_id": "string",
    "content": {
        "messages": [{
            "role": "user", "content": [{
                "type": "text", "text": "What are these images?"
            }]
        }]
    },
    "timestamp": "string",
    "user": {
        "name": "string", "email": "string"
    },
    "security": {
        "api_key": "string", "token": "string"
    },
    "error": {
        "code": "unknown_error", "message": "string", "details": "object"
    },
    "schema_version": "string"
}

system_response_token_message_with_text_content = {
    "type": "system_response_message",
    "id": "token_001",
    "thread_id": "thread_456",
    "parent_id": "id from user message",
    "content": {
        "text": "Response token can be json, code block or plain text"
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:02Z"
}
system_response_token_message_with_error_content = {
    "type": "error_message",
    "id": "token_001",
    "thread_id": "thread_456",
    "parent_id": "id from user message",
    "content": {
        "code": "unknown_error", "message": "ValidationError", "details": "The provided email format is invalid."
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:02Z"
}

user_interaction_response_message = {
    "type": "user_interaction_message",
    "id": "string",
    "thread_id": "string",
    "content": {
        "messages": [{
            "role": "user", "content": [{
                "type": "text", "text": "What are these images?"
            }]
        }]
    },
    "timestamp": "string",
    "user": {
        "name": "string", "email": "string"
    },
    "security": {
        "api_key": "string", "token": "string"
    },
    "error": {
        "code": "unknown_error", "message": "string", "details": "object"
    },
    "schema_version": "string"
}
system_intermediate_step_message = {
    "type": "system_intermediate_message",
    "id": "step_789",
    "thread_id": "thread_456",
    "parent_id": "id from user message",
    "intermediate_parent_id": "default",
    "content": {
        "name": "name of the step - example Query rephrasal",
        "payload": "Step information, it can be json or code block or it can be plain text"
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:01Z"
}

system_interaction_text_message = {
    "type": "system_interaction_message",
    "id": "interaction_303",
    "thread_id": "thread_456",
    "parent_id": "id from user message",
    "content": {
        "input_type": "text", "text": "Ask anything.", "placeholder": "What can you do?", "required": True
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:03Z"
}

system_interaction_binary_choice_message = {
    "type": "system_interaction_message",
    "id": "interaction_304",
    "thread_id": "thread_456",
    "parent_id": "msg_123",
    "content": {
        "input_type": "binary_choice",
        "text": "Should I continue or cancel?",
        "options": [{
            "id": "continue",
            "label": "Continue",
            "value": "continue",
        }, {
            "id": "cancel",
            "label": "Cancel",
            "value": "cancel",
        }],
        "required": True
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:03Z"
}

system_interaction_notification_message = {
    "type": "system_interaction_message",
    "id": "interaction_303",
    "thread_id": "thread_456",
    "parent_id": "id from user message",
    "content": {
        "input_type": "notification",
        "text": "Processing starting, it'll take some time",
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:03Z"
}

system_interaction_multiple_choice_radio_message = {
    "type": "system_interaction_message",
    "id": "interaction_305",
    "thread_id": "thread_456",
    "parent_id": "msg_123",
    "content": {
        "input_type": "radio",
        "text": "Please select your preferred notification method:",
        "options": [{
            "id": 'email', "label": "Email", "value": "email", "description": "Email notifications"
        }, {
            "id": 'sms', "label": "SMS", "value": "sms", "description": "SMS notifications"
        }, {
            "id": "push", "label": "Push Notification", "value": "push", "description": "Push notifications"
        }],
        "required": True
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:03Z"
}

system_interaction_multiple_choice_checkbox_message = {
    "type": "system_interaction_message",
    "id": "interaction_305",
    "thread_id": "thread_456",
    "parent_id": "msg_123",
    "content": {
        "input_type": "checkbox",
        "text": "Please select your preferred notification method:",
        "options": [{
            "id": 'email', "label": "Email", "value": "email", "description": "Email notifications"
        }, {
            "id": 'sms', "label": "SMS", "value": "sms", "description": "SMS notifications"
        }, {
            "id": "push", "label": "Push Notification", "value": "push", "description": "Push notifications"
        }],
        "required": True
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:03Z"
}
system_interaction_multiple_choice_dropdown_message = {
    "type": "system_interaction_message",
    "id": "interaction_305",
    "thread_id": "thread_456",
    "parent_id": "msg_123",
    "content": {
        "input_type": "dropdown",
        "text": "Please select your preferred notification method:",
        "options": [{
            "id": 'email', "label": "Email", "value": "email", "description": "Email notifications"
        }, {
            "id": 'sms', "label": "SMS", "value": "sms", "description": "SMS notifications"
        }, {
            "id": "push", "label": "Push Notification", "value": "push", "description": "Push notifications"
        }],
        "required": True
    },
    "status": "in_progress",
    "timestamp": "2025-01-13T10:00:03Z"
}


@pytest.fixture(scope="session", name="config")
def server_config(file_path: str = "tests/aiq/server/config.yml") -> BaseModel:
    data = None
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return Config(**data)


@pytest_asyncio.fixture(name="client")
async def client_fixture(config):
    app_config = load_config(config.app.config_filepath)
    front_end_worker = FastApiFrontEndPluginWorker(app_config)
    fastapi_app = front_end_worker.build_app()

    async with LifespanManager(fastapi_app) as manager:
        transport = ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url=f"http://{config.app.host}:{config.app.port}") as client:
            yield client


@pytest.mark.e2e
async def test_generate_endpoint(client: httpx.AsyncClient, config: Config):
    """Tests generate endpoint to verify it responds successfully."""
    input_message = {"input_message": f"{config.app.input}"}
    response = await client.post(f"{config.endpoint.generate}", json=input_message)
    assert response.status_code == 200


@pytest.mark.e2e
async def test_generate_stream_endpoint(client: httpx.AsyncClient, config: Config):
    """Tests generate stream endpoint to verify it responds successfully."""
    input_message = {"input_message": f"{config.app.input}"}
    response = await client.post(f"{config.endpoint.generate_stream}", json=input_message)
    assert response.status_code == 200


@pytest.mark.e2e
async def test_chat_endpoint(client: httpx.AsyncClient, config: Config):
    """Tests chat endpoint to verify it responds successfully."""
    input_message = {"messages": [{"role": "user", "content": f"{config.app.input}"}], "use_knowledge_base": True}
    response = await client.post(f"{config.endpoint.chat}", json=input_message)
    assert response.status_code == 200
    validated_response = AIQChatResponse(**response.json())
    assert isinstance(validated_response, AIQChatResponse)


@pytest.mark.e2e
async def test_chat_stream_endpoint(client: httpx.AsyncClient, config: Config):
    """Tests chat stream endpoint to verify it responds successfully."""
    input_message = {"messages": [{"role": "user", "content": f"{config.app.input}"}], "use_knowledge_base": True}
    response = await client.post(f"{config.endpoint.chat_stream}", json=input_message)
    assert response.status_code == 200
    data_match: re.Match[str] | None = re.search(r'data:\s*(.*)', response.text)
    data_match_dict: dict = json.loads(data_match.group(1))
    validated_response = AIQChatResponseChunk(**data_match_dict)
    assert isinstance(validated_response, AIQChatResponseChunk)


@pytest.mark.e2e
async def test_user_attributes_from_http_request(client: httpx.AsyncClient, config: Config):
    """Tests setting user attributes from HTTP request."""
    input_message = {"input_message": f"{config.app.input}"}
    headers = {"Header-Test": "application/json"}
    query_params = {"param1": "value1"}
    response = await client.post(
        f"{config.endpoint.generate}",
        json=input_message,
        headers=headers,
        params=query_params,
    )
    aiq_context = AIQContext.get()
    assert aiq_context.metadata.headers['header-test'] == headers["Header-Test"]
    assert aiq_context.metadata.query_params['param1'] == query_params["param1"]
    assert response.status_code == 200


async def test_valid_user_message():
    """Validate raw message against approved message type WebSocketUserMessage"""
    message_validator = MessageValidator()

    message = await message_validator.validate_message(user_message)
    assert isinstance(message, WebSocketUserMessage)


async def test_valid_system_response_token_message():
    """Validate raw message against approved message type WebSocketSystemResponseTokenMessage"""
    message_validator = MessageValidator()

    response_text_message = await message_validator.validate_message(system_response_token_message_with_text_content)
    response_error_message = await message_validator.validate_message(system_response_token_message_with_error_content)
    assert isinstance(response_text_message, WebSocketSystemResponseTokenMessage)
    assert isinstance(response_error_message, WebSocketSystemResponseTokenMessage)


async def test_valid_system_intermediate_step_message():
    """Validate raw message against approved message type WebSocketSystemIntermediateStepMessage"""
    message_validator = MessageValidator()

    intermediate_step_message = await message_validator.validate_message(system_intermediate_step_message)
    assert isinstance(intermediate_step_message, WebSocketSystemIntermediateStepMessage)


async def test_valid_user_interaction_response_message():
    """Validate raw message against approved message type WebSocketUserInteractionResponseMessage"""
    message_validator = MessageValidator()

    interaction_response_message = await message_validator.validate_message(user_interaction_response_message)
    assert isinstance(interaction_response_message, WebSocketUserInteractionResponseMessage)


valid_system_interaction_messages = [
    system_interaction_text_message,
    system_interaction_binary_choice_message,
    system_interaction_notification_message,
    system_interaction_multiple_choice_radio_message,
    system_interaction_multiple_choice_checkbox_message
]


@pytest.mark.parametrize("message", valid_system_interaction_messages)
async def test_valid_system_interaction_message(message):
    """Validate raw message against approved message type WebSocketSystemInteractionMessage"""
    message_validator = MessageValidator()

    system_interaction_message = await message_validator.validate_message(message)
    assert isinstance(system_interaction_message, WebSocketSystemInteractionMessage)


async def test_invalid_websocket_message():
    """Validate raw message against approved message type listed in (WebSocketMessageType)
    and return a system error response message with INVALID_MESSAGE error content if validation fails."""
    message_validator = MessageValidator()
    user_message["type"] = "invalid"
    message = await message_validator.validate_message(user_message)
    assert isinstance(message, WebSocketSystemResponseTokenMessage)
    assert message.content.code == ErrorTypes.INVALID_MESSAGE


aiq_response_payload_output_test = AIQResponsePayloadOutput(payload="TEST")
aiq_chat_response_test = AIQChatResponse(id="default",
                                         object="default",
                                         created=datetime.datetime.now(datetime.timezone.utc),
                                         choices=[AIQChoice(message=AIQChoiceMessage(), index=0)],
                                         usage=None)
aiq_chat_response_chunk_test = AIQChatResponseChunk(id="default",
                                                    choices=[AIQChoice(message=AIQChoiceMessage(), index=0)],
                                                    created=datetime.datetime.now(datetime.timezone.utc))
aiq_response_intermediate_step_test = AIQResponseIntermediateStep(id="default", name="default", payload="default")

validated_response_data_models = [
    aiq_response_payload_output_test, aiq_chat_response_test, aiq_chat_response_chunk_test
]


@pytest.mark.parametrize("data_model", validated_response_data_models)
async def test_resolve_response_message_type_by_input_data(data_model: BaseModel):
    """Resolve validated message type WebSocketMessageType.RESPONSE_MESSAGE from
    AIQResponsePayloadOutput, AIQChatResponse, AIQChatResponseChunk input data."""
    message_validator = MessageValidator()

    message_type = await message_validator.resolve_message_type_by_data(data_model)
    assert message_type == WebSocketMessageType.RESPONSE_MESSAGE


async def test_resolve_intermediate_step_message_type_by_input_data():
    """Resolve validated message type WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE from
    AIQResponseIntermediateStep input data."""
    message_validator = MessageValidator()

    message_type = await message_validator.resolve_message_type_by_data(aiq_response_intermediate_step_test)
    assert message_type == WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE


human_prompt_text_test = HumanPromptText(text="TEST", placeholder="TEST", required=True)
human_prompt_notification = HumanPromptNotification(text="TEST")
human_prompt_binary_choice_test = HumanPromptBinary(text="TEST",
                                                    options=[BinaryHumanPromptOption(), BinaryHumanPromptOption()])
human_prompt_radio_test = HumanPromptRadio(text="TEST", options=[MultipleChoiceOption()])
human_prompt_checkbox_test = HumanPromptCheckbox(text="TEST", options=[MultipleChoiceOption()])
human_prompt_dropdown_test = HumanPromptDropdown(text="TEST", options=[MultipleChoiceOption()])

validated_interaction_prompt_data_models = [
    human_prompt_text_test,
    human_prompt_notification,
    human_prompt_binary_choice_test,
    human_prompt_radio_test,
    human_prompt_checkbox_test,
    human_prompt_dropdown_test
]


@pytest.mark.parametrize("data_model", validated_interaction_prompt_data_models)
async def test_resolve_system_interaction_message_type_by_input_data(data_model: BaseModel):
    """Resolve validated message type WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE from
    HumanPromptBase input data."""
    message_validator = MessageValidator()

    message_type = await message_validator.resolve_message_type_by_data(data_model)
    assert message_type == WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE


async def test_resolve_error_message_type_by_invalid_input_data():
    """Resolve validated message type WebSocketMessageType.ERROR_MESSAGE from
    invalid input data."""
    message_validator = MessageValidator()

    message_type = await message_validator.resolve_message_type_by_data(TEST())
    assert message_type == WebSocketMessageType.ERROR_MESSAGE


async def test_aiq_response_to_websocket_message():
    """Tests AIQResponsePayloadOutput can be converted to a WebSocketSystemResponseTokenMessage"""
    message_validator = MessageValidator()

    aiq_response_content = await message_validator.convert_data_to_message_content(aiq_response_payload_output_test)

    aiq_response_to_system_response = await message_validator.create_system_response_token_message(
        message_id="TEST", parent_id="TEST", content=aiq_response_content, status="in_progress")

    assert isinstance(aiq_response_content, SystemResponseContent)
    assert isinstance(aiq_response_to_system_response, WebSocketSystemResponseTokenMessage)


async def test_aiq_chat_response_to_websocket_message():
    """Tests AIQChatResponse can be converted to a WebSocketSystemResponseTokenMessage"""
    message_validator = MessageValidator()

    aiq_chat_response_content = await message_validator.convert_data_to_message_content(aiq_chat_response_test)

    aiq_chat_response_to_system_response = await message_validator.create_system_response_token_message(
        message_id="TEST", parent_id="TEST", content=aiq_chat_response_content, status="in_progress")

    assert isinstance(aiq_chat_response_content, SystemResponseContent)
    assert isinstance(aiq_chat_response_to_system_response, WebSocketSystemResponseTokenMessage)


async def test_chat_response_chunk_to_websocket_message():
    """Tests AIQChatResponseChunk can be converted to a WebSocketSystemResponseTokenMessage"""
    message_validator = MessageValidator()

    aiq_chat_repsonse_chunk_content = await message_validator.convert_data_to_message_content(
        aiq_chat_response_chunk_test)

    aiq_chat_repsonse_chunk_to_system_response = await message_validator.create_system_response_token_message(
        message_id="TEST", parent_id="TEST", content=aiq_chat_repsonse_chunk_content, status="in_progress")

    assert isinstance(aiq_chat_repsonse_chunk_content, SystemResponseContent)
    assert isinstance(aiq_chat_repsonse_chunk_to_system_response, WebSocketSystemResponseTokenMessage)


async def test_aiq_intermediate_step_to_websocket_message():
    """Tests AIQResponseIntermediateStep can be converted to a WebSocketSystemIntermediateStepMessage"""
    message_validator = MessageValidator()

    aiq_intermediate_step_content = await message_validator.convert_data_to_message_content(
        aiq_response_intermediate_step_test)

    intermediate_step_content_to_message = await message_validator.create_system_intermediate_step_message(
        message_id="TEST", parent_id="TEST", content=aiq_intermediate_step_content, status="in_progress")

    assert isinstance(aiq_intermediate_step_content, SystemIntermediateStepContent)
    assert isinstance(intermediate_step_content_to_message, WebSocketSystemIntermediateStepMessage)


async def test_text_prompt_to_websocket_message_to_text_response():
    message_validator = MessageValidator()

    human_text_content = await message_validator.convert_data_to_message_content(human_prompt_text_test)

    human_text_to_interaction_message = await message_validator.create_system_interaction_message(
        message_id="TEST", parent_id="TEST", content=human_text_content, status="in_progress")

    human_text_response_content = await message_validator.convert_text_content_to_human_response(
        TextContent(), human_text_content)

    assert isinstance(human_text_content, HumanPromptText)
    assert isinstance(human_text_to_interaction_message, WebSocketSystemInteractionMessage)
    assert isinstance(human_text_to_interaction_message.content, HumanPromptText)
    assert isinstance(human_text_response_content, HumanResponseText)


async def test_binary_choice_prompt_to_websocket_message_to_binary_choice_response():
    message_validator = MessageValidator()

    human_binary_choice_content = await message_validator.convert_data_to_message_content(
        human_prompt_binary_choice_test)

    human_binary_choice_to_interaction_message = await message_validator.create_system_interaction_message(
        message_id="TEST", parent_id="TEST", content=human_binary_choice_content, status="in_progress")

    human_text_response_content = await message_validator.convert_text_content_to_human_response(
        TextContent(), human_binary_choice_content)

    assert isinstance(human_binary_choice_content, HumanPromptBinary)
    assert isinstance(human_binary_choice_to_interaction_message, WebSocketSystemInteractionMessage)
    assert isinstance(human_binary_choice_to_interaction_message.content, HumanPromptBinary)
    assert isinstance(human_text_response_content, HumanResponseBinary)


async def test_radio_choice_prompt_to_websocket_message_to_radio_choice_response():
    message_validator = MessageValidator()

    human_radio_choice_content = await message_validator.convert_data_to_message_content(human_prompt_radio_test)

    human_radio_choice_to_interaction_message = await message_validator.create_system_interaction_message(
        message_id="TEST", parent_id="TEST", content=human_radio_choice_content, status="in_progress")

    human_radio_response_content = await message_validator.convert_text_content_to_human_response(
        TextContent(), human_radio_choice_content)

    assert isinstance(human_radio_choice_content, HumanPromptRadio)
    assert isinstance(human_radio_choice_to_interaction_message, WebSocketSystemInteractionMessage)
    assert isinstance(human_radio_choice_to_interaction_message.content, HumanPromptRadio)
    assert isinstance(human_radio_response_content, HumanResponseRadio)


async def test_dropdown_choice_prompt_to_websocket_message_to_dropdown_choice_response():
    message_validator = MessageValidator()

    human_dropdown_choice_content = await message_validator.convert_data_to_message_content(human_prompt_dropdown_test)

    human_dropdown_choice_to_interaction_message = await message_validator.create_system_interaction_message(
        message_id="TEST", parent_id="TEST", content=human_dropdown_choice_content, status="in_progress")

    human_dropdown_response_content = await message_validator.convert_text_content_to_human_response(
        TextContent(), human_dropdown_choice_content)

    assert isinstance(human_dropdown_choice_content, HumanPromptDropdown)
    assert isinstance(human_dropdown_choice_to_interaction_message, WebSocketSystemInteractionMessage)
    assert isinstance(human_dropdown_choice_to_interaction_message.content, HumanPromptDropdown)
    assert isinstance(human_dropdown_response_content, HumanResponseDropdown)


async def test_checkbox_choice_prompt_to_websocket_message_to_checkbox_choice_response():
    message_validator = MessageValidator()

    human_checkbox_choice_content = await message_validator.convert_data_to_message_content(human_prompt_checkbox_test)

    human_checkbox_choice_to_interaction_message = await message_validator.create_system_interaction_message(
        message_id="TEST", parent_id="TEST", content=human_checkbox_choice_content, status="in_progress")

    human_checkbox_response_content = await message_validator.convert_text_content_to_human_response(
        TextContent(), human_checkbox_choice_content)

    assert isinstance(human_checkbox_choice_content, HumanPromptCheckbox)
    assert isinstance(human_checkbox_choice_to_interaction_message, WebSocketSystemInteractionMessage)
    assert isinstance(human_checkbox_choice_to_interaction_message.content, HumanPromptCheckbox)
    assert isinstance(human_checkbox_response_content, HumanResponseCheckbox)


async def test_websocket_error_message():
    message_validator = MessageValidator()

    try:
        invalid_message_type = "invalid_message_type"
        invalid_data_model = TEST()
        message_schema: type[BaseModel] = await message_validator.get_message_schema_by_type(invalid_message_type)

        content: BaseModel = await message_validator.convert_data_to_message_content(invalid_data_model)

        if (issubclass(message_schema, Error)):
            raise TypeError(f"TESTING MESSAGE ERROR PATH: {content}")

        if (isinstance(content, Error)):
            raise ValidationError(f"TESTING MESSAGE ERROR PATH: {content}")

    except (ValidationError, TypeError, ValueError) as e:
        message = await message_validator.create_system_response_token_message(
            message_type=WebSocketMessageType.ERROR_MESSAGE,
            content=Error(code=ErrorTypes.UNKNOWN_ERROR, message="Test message", details=str(e)))

        assert isinstance(message, WebSocketSystemResponseTokenMessage)


async def test_valid_openai_chat_request_fields():
    """Test that AIQChatRequest accepts valid field structures"""
    # Test with minimal required fields
    minimal_request = {"messages": [{"role": "user", "content": "Hello"}]}

    # Test with comprehensive valid fields
    comprehensive_request = {
        "messages": [{
            "role": "user", "content": "Hello"
        }],
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "stream": False,
        "stop": ["END"],
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "n": 1,
        "user": "test_user",
        "use_knowledge_base": True,  # Test extra fields are allowed
        "custom_field": "should_be_allowed",
        "another_custom": {
            "nested": "value"
        }
    }

    # Both should validate successfully
    assert AIQChatRequest(**minimal_request)
    assert AIQChatRequest(**comprehensive_request)


async def test_invalid_openai_chat_request_fields():
    """Test that AIQChatRequest raises ValidationError for improper payloads"""

    with pytest.raises(ValidationError):
        AIQChatRequest()

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=[{"content": "Hello"}])

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=[{"role": "user"}])

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=[{"role": "user", "content": "Hello"}], temperature="not_a_number")

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=[{"role": "user", "content": "Hello"}], max_tokens="not_an_integer")

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=[{"role": "user", "content": "Hello"}], stream="not_a_boolean")

    with pytest.raises(ValidationError):
        AIQChatRequest(messages="not_a_list")

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=["not_a_dict"])

    with pytest.raises(ValidationError):
        AIQChatRequest(messages=None)
