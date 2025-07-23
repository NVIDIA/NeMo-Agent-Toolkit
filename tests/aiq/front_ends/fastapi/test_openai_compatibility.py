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

from contextlib import asynccontextmanager

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from httpx import AsyncClient
from httpx_sse import aconnect_sse

from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQChoiceDelta
from aiq.data_models.api_server import Message
from aiq.data_models.config import AIQConfig
from aiq.data_models.config import GeneralConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from aiq.test.functions import EchoFunctionConfig
from aiq.test.functions import StreamingEchoFunctionConfig


@asynccontextmanager
async def _build_client(config: AIQConfig,
                        worker_class: type[FastApiFrontEndPluginWorker] = FastApiFrontEndPluginWorker):
    """Helper to build test client with proper lifecycle management"""
    worker = worker_class(config)
    app = worker.build_app()

    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


def test_fastapi_config_openai_api_v1_path_field():
    """Test that openai_api_v1_path field is properly added to config"""
    # Test default value (None)
    config = FastApiFrontEndConfig.EndpointBase(method="POST", description="test")
    assert hasattr(config, 'openai_api_v1_path')
    assert config.openai_api_v1_path is None

    # Test explicit path
    config = FastApiFrontEndConfig.EndpointBase(method="POST",
                                                description="test",
                                                openai_api_v1_path="/v1/chat/completions")
    assert config.openai_api_v1_path == "/v1/chat/completions"

    # Test explicit None
    config = FastApiFrontEndConfig.EndpointBase(method="POST", description="test", openai_api_v1_path=None)
    assert config.openai_api_v1_path is None


def test_aiq_chat_request_openai_fields():
    """Test that AIQChatRequest includes all OpenAI Chat Completions API fields"""
    # Test with minimal required fields
    request = AIQChatRequest(messages=[Message(content="Hello", role="user")])
    assert request.messages[0].content == "Hello"
    assert request.stream is False  # Default value

    # Test with all OpenAI fields
    request = AIQChatRequest(messages=[Message(content="Hello", role="user")],
                             model="gpt-3.5-turbo",
                             frequency_penalty=0.5,
                             logit_bias={"token1": 0.1},
                             logprobs=True,
                             top_logprobs=5,
                             max_tokens=100,
                             n=1,
                             presence_penalty=-0.5,
                             response_format={"type": "json_object"},
                             seed=42,
                             service_tier="auto",
                             stop=["END"],
                             stream=True,
                             stream_options={"include_usage": True},
                             temperature=0.7,
                             top_p=0.9,
                             tools=[{
                                 "type": "function", "function": {
                                     "name": "test"
                                 }
                             }],
                             tool_choice="auto",
                             parallel_tool_calls=False,
                             user="user123")

    # Verify all fields are set correctly
    assert request.model == "gpt-3.5-turbo"
    assert request.frequency_penalty == 0.5
    assert request.logit_bias == {"token1": 0.1}
    assert request.logprobs is True
    assert request.top_logprobs == 5
    assert request.max_tokens == 100
    assert request.n == 1
    assert request.presence_penalty == -0.5
    assert request.response_format == {"type": "json_object"}
    assert request.seed == 42
    assert request.service_tier == "auto"
    assert request.stop == ["END"]
    assert request.stream is True
    assert request.stream_options == {"include_usage": True}
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.tools == [{"type": "function", "function": {"name": "test"}}]
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is False
    assert request.user == "user123"


def test_aiq_choice_delta_class():
    """Test that AIQChoiceDelta class works correctly"""
    # Test empty delta
    delta = AIQChoiceDelta()
    assert delta.content is None
    assert delta.role is None

    # Test delta with content
    delta = AIQChoiceDelta(content="Hello")
    assert delta.content == "Hello"
    assert delta.role is None

    # Test delta with role
    delta = AIQChoiceDelta(role="assistant")
    assert delta.content is None
    assert delta.role == "assistant"

    # Test delta with both
    delta = AIQChoiceDelta(content="Hello", role="assistant")
    assert delta.content == "Hello"
    assert delta.role == "assistant"


def test_aiq_chat_response_chunk_create_streaming_chunk():
    """Test the new create_streaming_chunk method"""
    # Test basic streaming chunk
    chunk = AIQChatResponseChunk.create_streaming_chunk(content="Hello", role="assistant")

    assert chunk.choices[0].delta.content == "Hello"
    assert chunk.choices[0].delta.role == "assistant"
    assert chunk.choices[0].message is None
    assert chunk.choices[0].finish_reason is None
    assert chunk.object == "chat.completion.chunk"

    # Test streaming chunk with finish_reason
    chunk = AIQChatResponseChunk.create_streaming_chunk(content="", finish_reason="stop")

    assert chunk.choices[0].delta.content == ""
    assert chunk.choices[0].finish_reason == "stop"


def test_aiq_chat_response_timestamp_serialization():
    """Test that timestamps are serialized as Unix timestamps for OpenAI compatibility"""
    import datetime

    # Create response with known timestamp
    test_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    response = AIQChatResponse.from_string("Hello", created=test_time)

    # Serialize to JSON
    json_data = response.model_dump()

    # Verify timestamp is Unix timestamp (1704110400 = 2024-01-01 12:00:00 UTC)
    assert json_data["created"] == 1704110400

    # Same test for chunk
    chunk = AIQChatResponseChunk.from_string("Hello", created=test_time)
    chunk_json = chunk.model_dump()
    assert chunk_json["created"] == 1704110400


@pytest.mark.parametrize("openai_api_v1_path", ["/v1/chat/completions", None])
async def test_legacy_vs_openai_v1_mode_endpoints(openai_api_v1_path: str | None):
    """Test that endpoints are created correctly for both legacy and OpenAI v1 compatible modes"""

    # Configure with the specified mode
    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = openai_api_v1_path
    front_end_config.workflow.openai_api_path = "/v1/chat/completions"

    config = AIQConfig(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=True),
    )

    async with _build_client(config) as client:
        base_path = "/v1/chat/completions"

        if openai_api_v1_path:
            # OpenAI v1 Compatible Mode: single endpoint handles both streaming and non-streaming

            # Test non-streaming request
            response = await client.post(base_path,
                                         json={
                                             "messages": [{
                                                 "content": "Hello", "role": "user"
                                             }], "stream": False
                                         })
            assert response.status_code == 200
            chat_response = AIQChatResponse.model_validate(response.json())
            assert chat_response.choices[0].message.content == "Hello"
            assert chat_response.object == "chat.completion"

            # Test streaming request
            response_chunks = []
            async with aconnect_sse(client,
                                    "POST",
                                    base_path,
                                    json={
                                        "messages": [{
                                            "content": "World", "role": "user"
                                        }], "stream": True
                                    }) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.data != "[DONE]":
                        chunk = AIQChatResponseChunk.model_validate(sse.json())
                        response_chunks.append(chunk)

            assert event_source.response.status_code == 200
            assert len(response_chunks) > 0
            # In OpenAI compatible mode, we should get proper streaming response
            # The chunks use the existing streaming infrastructure format
            has_content = any((chunk.choices[0].message and chunk.choices[0].message.content) or (
                chunk.choices[0].delta and chunk.choices[0].delta.content) for chunk in response_chunks)
            assert has_content

        else:
            # Legacy Mode: separate endpoints for streaming and non-streaming

            # Test non-streaming endpoint (base path)
            response = await client.post(base_path, json={"messages": [{"content": "Hello", "role": "user"}]})
            assert response.status_code == 200
            chat_response = AIQChatResponse.model_validate(response.json())
            assert chat_response.choices[0].message.content == "Hello"

            # Test streaming endpoint (base path + /stream)
            response_chunks = []
            async with aconnect_sse(client,
                                    "POST",
                                    f"{base_path}/stream",
                                    json={"messages": [{
                                        "content": "World", "role": "user"
                                    }]}) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.data != "[DONE]":
                        chunk = AIQChatResponseChunk.model_validate(sse.json())
                        response_chunks.append(chunk)

            assert event_source.response.status_code == 200
            assert len(response_chunks) > 0
            # In legacy mode, chunks should use message field
            has_message_content = any(chunk.choices[0].message and chunk.choices[0].message.content
                                      for chunk in response_chunks)
            assert has_message_content


async def test_openai_compatible_mode_stream_parameter():
    """Test that OpenAI compatible mode correctly handles stream parameter"""

    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = "/v1/chat/completions"
    front_end_config.workflow.openai_api_path = "/v1/chat/completions"

    # Use streaming config since that's what's available
    config = AIQConfig(
        general=GeneralConfig(front_end=front_end_config),
        workflow=StreamingEchoFunctionConfig(use_openai_api=True),
    )

    async with _build_client(config) as client:
        base_path = "/v1/chat/completions"

        # Test stream=true (should return streaming response)
        # This is the main functionality we're testing - single endpoint routing
        async with aconnect_sse(client,
                                "POST",
                                base_path,
                                json={
                                    "messages": [{
                                        "content": "Hello", "role": "user"
                                    }], "stream": True
                                }) as event_source:
            chunks_received = 0
            async for sse in event_source.aiter_sse():
                if sse.data != "[DONE]":
                    chunk = AIQChatResponseChunk.model_validate(sse.json())
                    assert chunk.object == "chat.completion.chunk"
                    chunks_received += 1
                    if chunks_received >= 2:  # Stop after receiving a few chunks
                        break

        assert event_source.response.status_code == 200
        assert event_source.response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_legacy_mode_backward_compatibility():
    """Test that legacy mode maintains exact backward compatibility"""

    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = None  # Legacy mode
    front_end_config.workflow.openai_api_path = "/v1/chat/completions"

    config = AIQConfig(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=True),
    )

    async with _build_client(config) as client:
        base_path = "/v1/chat/completions"

        # Test legacy non-streaming endpoint structure
        response = await client.post(base_path, json={"messages": [{"content": "Hello", "role": "user"}]})
        assert response.status_code == 200
        chat_response = AIQChatResponse.model_validate(response.json())

        # Verify legacy response structure
        assert chat_response.choices[0].message is not None
        assert chat_response.choices[0].message.content == "Hello"
        assert chat_response.object == "chat.completion"

        # Test legacy streaming endpoint structure
        response_chunks = []
        async with aconnect_sse(client,
                                "POST",
                                f"{base_path}/stream",
                                json={"messages": [{
                                    "content": "World", "role": "user"
                                }]}) as event_source:
            async for sse in event_source.aiter_sse():
                if sse.data != "[DONE]":
                    chunk = AIQChatResponseChunk.model_validate(sse.json())
                    response_chunks.append(chunk)
                    if len(response_chunks) >= 1:  # Just need to verify structure
                        break

        assert event_source.response.status_code == 200
        assert len(response_chunks) > 0

        # Verify legacy chunk structure (uses message, not delta)
        chunk = response_chunks[0]
        assert chunk.choices[0].message is not None
        assert chunk.choices[0].message.content == "World"
        assert chunk.object == "chat.completion.chunk"
        # In legacy mode, delta should not be populated
        assert chunk.choices[0].delta is None or (chunk.choices[0].delta.content is None
                                                  and chunk.choices[0].delta.role is None)


def test_openai_request_validation():
    """Test that OpenAI request validation works correctly"""

    # Test valid ranges
    valid_request = AIQChatRequest(messages=[Message(content="Hello", role="user")],
                                   temperature=1.0,
                                   top_p=0.5,
                                   frequency_penalty=1.0,
                                   presence_penalty=-1.0,
                                   max_tokens=100,
                                   n=5,
                                   top_logprobs=10)
    assert valid_request.temperature == 1.0
    assert valid_request.top_p == 0.5

    # Test invalid temperature (should raise validation error)
    with pytest.raises(ValueError):
        AIQChatRequest(
            messages=[Message(content="Hello", role="user")],
            temperature=3.0  # Invalid: > 2.0
        )

    # Test invalid frequency_penalty (should raise validation error)
    with pytest.raises(ValueError):
        AIQChatRequest(
            messages=[Message(content="Hello", role="user")],
            frequency_penalty=3.0  # Invalid: > 2.0
        )

    # Test invalid top_p (should raise validation error)
    with pytest.raises(ValueError):
        AIQChatRequest(
            messages=[Message(content="Hello", role="user")],
            top_p=1.5  # Invalid: > 1.0
        )

    # Test invalid max_tokens (should raise validation error)
    with pytest.raises(ValueError):
        AIQChatRequest(
            messages=[Message(content="Hello", role="user")],
            max_tokens=0  # Invalid: < 1
        )

    # Test invalid n (should raise validation error)
    with pytest.raises(ValueError):
        AIQChatRequest(
            messages=[Message(content="Hello", role="user")],
            n=200  # Invalid: > 128
        )

    # Test invalid top_logprobs (should raise validation error)
    with pytest.raises(ValueError):
        AIQChatRequest(
            messages=[Message(content="Hello", role="user")],
            top_logprobs=25  # Invalid: > 20
        )


def test_converter_functions_backward_compatibility():
    """Test that converter functions handle both legacy and new formats"""
    from aiq.data_models.api_server import _aiq_chat_response_chunk_to_string
    from aiq.data_models.api_server import _chat_response_to_chat_response_chunk

    # Test legacy chunk (with message) conversion to string
    legacy_chunk = AIQChatResponseChunk.from_string("Legacy content")
    legacy_content = _aiq_chat_response_chunk_to_string(legacy_chunk)
    assert legacy_content == "Legacy content"

    # Test new chunk (with delta) conversion to string
    new_chunk = AIQChatResponseChunk.create_streaming_chunk("New content")
    new_content = _aiq_chat_response_chunk_to_string(new_chunk)
    assert new_content == "New content"

    # Test response to chunk conversion preserves message structure
    response = AIQChatResponse.from_string("Response content")
    converted_chunk = _chat_response_to_chat_response_chunk(response)

    # Should preserve original message structure for backward compatibility
    assert converted_chunk.choices[0].message is not None
    assert converted_chunk.choices[0].message.content == "Response content"
