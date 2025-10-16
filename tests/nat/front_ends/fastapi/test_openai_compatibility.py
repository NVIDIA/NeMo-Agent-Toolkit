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

import pytest
from httpx_sse import aconnect_sse

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import Message
from nat.data_models.api_server import Usage
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.test.functions import EchoFunctionConfig
from nat.test.functions import StreamingEchoFunctionConfig
from nat.test.utils import build_nat_client


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


def test_nat_chat_request_openai_fields():
    """Test that ChatRequest includes all OpenAI Chat Completions API fields"""
    # Test with minimal required fields
    request = ChatRequest(messages=[Message(content="Hello", role="user")])
    assert request.messages[0].content == "Hello"
    assert request.stream is False  # Default value

    # Test with all OpenAI fields
    request = ChatRequest(messages=[Message(content="Hello", role="user")],
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


def test_nat_choice_delta_class():
    """Test that ChoiceDelta class works correctly"""
    # Test empty delta
    delta = ChoiceDelta()
    assert delta.content is None
    assert delta.role is None

    # Test delta with content
    delta = ChoiceDelta(content="Hello")
    assert delta.content == "Hello"
    assert delta.role is None

    # Test delta with role
    delta = ChoiceDelta(role="assistant")
    assert delta.content is None
    assert delta.role == "assistant"

    # Test delta with both
    delta = ChoiceDelta(content="Hello", role="assistant")
    assert delta.content == "Hello"
    assert delta.role == "assistant"


def test_nat_chat_response_chunk_create_streaming_chunk():
    """Test the new create_streaming_chunk method"""
    # Test basic streaming chunk
    chunk = ChatResponseChunk.create_streaming_chunk(content="Hello", role=UserMessageContentRoleType.ASSISTANT)

    assert chunk.choices[0].delta.content == "Hello"
    assert chunk.choices[0].delta.role == UserMessageContentRoleType.ASSISTANT
    assert chunk.choices[0].finish_reason is None
    assert chunk.object == "chat.completion.chunk"

    # Test streaming chunk with finish_reason
    chunk = ChatResponseChunk.create_streaming_chunk(content="", finish_reason="stop")

    assert chunk.choices[0].delta.content == ""
    assert chunk.choices[0].finish_reason == "stop"


def test_nat_chat_response_timestamp_serialization():
    """Test that timestamps are serialized as Unix timestamps for OpenAI compatibility"""
    import datetime

    # Create response with known timestamp
    test_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    # Create usage statistics for test
    usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response = ChatResponse.from_string("Hello", created=test_time, usage=usage)

    # Serialize to JSON
    json_data = response.model_dump()

    # Verify timestamp is Unix timestamp (1704110400 = 2024-01-01 12:00:00 UTC)
    assert json_data["created"] == 1704110400

    # Same test for chunk
    chunk = ChatResponseChunk.from_string("Hello", created=test_time)
    chunk_json = chunk.model_dump()
    assert chunk_json["created"] == 1704110400


@pytest.mark.parametrize("openai_api_v1_path", ["/v1/chat/completions", None])
async def test_legacy_vs_openai_v1_mode_endpoints(openai_api_v1_path: str | None):
    """Test that endpoints are created correctly for both legacy and OpenAI v1 compatible modes"""

    # Configure with the specified mode
    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = openai_api_v1_path
    front_end_config.workflow.openai_api_path = "/v1/chat/completions"

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=True),
    )

    async with build_nat_client(config) as client:
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
            chat_response = ChatResponse.model_validate(response.json())
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
                        chunk = ChatResponseChunk.model_validate(sse.json())
                        response_chunks.append(chunk)

            assert event_source.response.status_code == 200
            assert len(response_chunks) > 0

        else:
            # Legacy Mode: separate endpoints for streaming and non-streaming

            # Test non-streaming endpoint (base path)
            response = await client.post(base_path, json={"messages": [{"content": "Hello", "role": "user"}]})
            assert response.status_code == 200
            chat_response = ChatResponse.model_validate(response.json())
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
                        chunk = ChatResponseChunk.model_validate(sse.json())
                        response_chunks.append(chunk)

            assert event_source.response.status_code == 200
            assert len(response_chunks) > 0


async def test_openai_compatible_mode_stream_parameter():
    """Test that OpenAI compatible mode correctly handles stream parameter"""

    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = "/v1/chat/completions"
    front_end_config.workflow.openai_api_path = "/v1/chat/completions"

    # Use streaming config since that's what's available
    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=StreamingEchoFunctionConfig(use_openai_api=True),
    )

    async with build_nat_client(config) as client:
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
                    chunk = ChatResponseChunk.model_validate(sse.json())
                    assert chunk.object == "chat.completion.chunk"
                    chunks_received += 1
                    if chunks_received >= 2:  # Stop after receiving a few chunks
                        break

        assert event_source.response.status_code == 200
        assert event_source.response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_legacy_non_streaming_response_format():
    """Test non-streaming legacy endpoint response format matches exact OpenAI structure"""

    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_path = "/chat"

    # Use EchoFunctionConfig with specific content to match expected response
    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=True),
    )

    async with build_nat_client(config) as client:
        # Send request to legacy OpenAI endpoint
        response = await client.post("/chat",
                                     json={
                                         "messages": [{
                                             "role": "user", "content": "Hello! How can I assist you today?"
                                         }],
                                         "stream": False
                                     })

        assert response.status_code == 200
        data = response.json()

        # Validate response structure exactly matches OpenAI ChatCompletion format
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) == 1

        # Verify choices array structure (OpenAI spec: array of choice objects)
        choice = data["choices"][0]

        # Essential choice fields per OpenAI spec
        assert choice["index"] == 0, "Choice index should be 0 for single completion"
        assert "message" in choice, "Choice must contain message object"
        assert "finish_reason" in choice, "Choice must contain finish_reason"

        # Message structure validation
        message = choice["message"]
        assert "role" in message, "Message must contain role"
        assert message["role"] == "assistant", "Response message role should be assistant"
        assert "content" in message, "Message must contain content"
        assert isinstance(message["content"], str), "Message content must be string"

        # Finish reason validation
        finish_reason = choice["finish_reason"]
        valid_finish_reasons = {"stop", "length", "content_filter", "tool_calls", "function_call"}
        assert finish_reason in valid_finish_reasons, f"Invalid finish_reason: {finish_reason}"

        # Usage validation (OpenAI spec requires usage field for non-streaming)
        assert "usage" in data, "Non-streaming response must include usage"
        usage = data["usage"]
        assert "prompt_tokens" in usage, "Usage must include prompt_tokens"
        assert "completion_tokens" in usage, "Usage must include completion_tokens"
        assert "total_tokens" in usage, "Usage must include total_tokens"

        # Validate token counts are non-negative integers
        assert isinstance(usage["prompt_tokens"], int), "prompt_tokens must be integer"
        assert isinstance(usage["completion_tokens"], int), "completion_tokens must be integer"
        assert isinstance(usage["total_tokens"], int), "total_tokens must be integer"
        assert usage["prompt_tokens"] >= 0, "prompt_tokens must be non-negative"
        assert usage["completion_tokens"] >= 0, "completion_tokens must be non-negative"
        assert usage["total_tokens"] >= 0, "total_tokens must be non-negative"

        # Validate total_tokens = prompt_tokens + completion_tokens
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], \
            "total_tokens must equal prompt_tokens + completion_tokens"


async def test_legacy_streaming_response_format():
    """
    Validate only the required structural shape of legacy streaming
    (/chat/stream).
    """
    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_path = "/chat"

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=StreamingEchoFunctionConfig(use_openai_api=True),
    )

    async with build_nat_client(config) as client:
        async with aconnect_sse(client,
                                "POST",
                                "/chat/stream",
                                json={
                                    "messages": [{
                                        "role": "user", "content": "Hello"
                                    }], "stream": True
                                }) as event_source:

            chunks = []
            async for sse in event_source.aiter_sse():
                if sse.data == "[DONE]":
                    break
                chunks.append(sse.json())

            # Transport-level checks
            assert event_source.response.status_code == 200
            ct = event_source.response.headers.get("content-type", "")
            assert ct.startswith("text/event-stream"), f"Unexpected Content-Type: {ct}"
            assert len(chunks) > 0, "Expected at least one JSON chunk before [DONE]"

    # ---- Structural validation of chunks ----
    valid_final_reason_seen = False
    valid_finish_reasons = {"stop", "length", "content_filter", "tool_calls", "function_call"}

    for i, chunk in enumerate(chunks):
        # Required root fields for a streaming chunk
        assert chunk.get("object") == "chat.completion.chunk", f"Chunk {i}: wrong object"
        assert chunk.get("id"), f"Chunk {i}: missing id"
        assert "created" in chunk, f"Chunk {i}: missing created"
        assert chunk.get("model"), f"Chunk {i}: missing model"
        assert "choices" in chunk, f"Chunk {i}: missing choices"

        # choices can be empty on a usage-only summary chunk
        if not chunk["choices"]:
            continue

        for c_idx, choice in enumerate(chunk["choices"]):
            # Required choice fields in streaming
            assert "index" in choice, f"Chunk {i} choice {c_idx}: missing index"
            assert "delta" in choice, f"Chunk {i} choice {c_idx}: missing delta"
            # Must NOT include full message in streaming
            assert "message" not in choice, f"Chunk {i} choice {c_idx}: message must not appear in streaming"
            # finish_reason must exist; may be null until final chunk
            assert "finish_reason" in choice, f"Chunk {i} choice {c_idx}: missing finish_reason"

            fr = choice.get("finish_reason")
            if fr is not None:
                assert fr in valid_finish_reasons, f"Chunk {i} choice {c_idx}: invalid finish_reason {fr}"
                valid_final_reason_seen = True

    # At least one non-null finish_reason should appear across the stream (finalization)
    assert valid_final_reason_seen, "Expected a final chunk with non-null finish_reason"


async def test_openai_compatible_non_streaming_response_format():
    """Test non-streaming OpenAI compatible endpoint response format matches exact OpenAI structure"""

    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = "/v1/chat/completions"

    # Use EchoFunctionConfig with specific content to match expected response
    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=EchoFunctionConfig(use_openai_api=True),
    )

    async with build_nat_client(config) as client:
        # Send request to actual OpenAI endpoint - this will trigger generate_single_response
        response = await client.post("/v1/chat/completions",
                                     json={
                                         "messages": [{
                                             "role": "user", "content": "Hello! How can I assist you today?"
                                         }],
                                         "stream": False
                                     })

        assert response.status_code == 200
        data = response.json()

        # Validate response structure exactly matches OpenAI ChatCompletion format
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) == 1

        # Verify choices array structure (OpenAI spec: array of choice objects)
        choice = data["choices"][0]

        # Essential choice fields per OpenAI spec
        assert choice["index"] == 0, "Choice index should be 0 for single completion"
        assert isinstance(choice["index"], int), "Choice index should be integer"

        # finish_reason: stop|length|content_filter|tool_calls|function_call
        assert choice["finish_reason"] == "stop", "Finish reason should be 'stop' for completed response"
        assert choice["finish_reason"] in ["stop", "length", "content_filter", "tool_calls", "function_call"], \
            f"Invalid finish_reason: {choice['finish_reason']}"

        # Message object should be present for non-streaming, delta should not
        assert "message" in choice, "Non-streaming response must have message field"
        assert "delta" not in choice, "Non-streaming response should not have delta field"

        # OpenAI spec requires logprobs field (can be null if not requested)
        if "logprobs" in choice:
            # logprobs can be null or object with content/refusal arrays
            assert choice["logprobs"] is None or isinstance(choice["logprobs"], dict)

        # Verify message object structure per OpenAI spec
        message = choice["message"]

        # Essential message fields
        assert "role" in message, "Message must have role field"
        assert message["role"] == "assistant", f"Expected assistant role, got: {message['role']}"
        assert "content" in message, "Message must have content field"
        assert message["content"] == "Hello! How can I assist you today?", "Echo function should return input content"
        assert isinstance(message["content"], str), "Message content should be string"

        # Verify usage statistics per OpenAI spec
        assert "usage" in data, "Response must include usage statistics"
        usage = data["usage"]

        # Essential usage fields
        assert "prompt_tokens" in usage, "Usage must include prompt_tokens"
        assert "completion_tokens" in usage, "Usage must include completion_tokens"
        assert "total_tokens" in usage, "Usage must include total_tokens"


async def test_openai_compatible_streaming_response_format():
    """
    Validate only the required structural shape of OpenAI-compatible streaming
    (/v1/chat/completions with stream=True).
    """
    front_end_config = FastApiFrontEndConfig()
    front_end_config.workflow.openai_api_v1_path = "/v1/chat/completions"

    config = Config(
        general=GeneralConfig(front_end=front_end_config),
        workflow=StreamingEchoFunctionConfig(use_openai_api=True),
    )

    async with build_nat_client(config) as client:
        async with aconnect_sse(client,
                                "POST",
                                "/v1/chat/completions",
                                json={
                                    "messages": [{
                                        "role": "user", "content": "Hello"
                                    }], "stream": True
                                }) as event_source:

            chunks = []
            async for sse in event_source.aiter_sse():
                if sse.data == "[DONE]":
                    break
                chunks.append(sse.json())

            # Transport-level checks
            assert event_source.response.status_code == 200
            ct = event_source.response.headers.get("content-type", "")
            assert ct.startswith("text/event-stream"), f"Unexpected Content-Type: {ct}"
            assert len(chunks) > 0, "Expected at least one JSON chunk before [DONE]"

    # ---- Structural validation of chunks ----
    valid_final_reason_seen = False
    valid_finish_reasons = {"stop", "length", "content_filter", "tool_calls", "function_call"}

    for i, chunk in enumerate(chunks):
        # Required root fields for a streaming chunk
        assert chunk.get("object") == "chat.completion.chunk", f"Chunk {i}: wrong object"
        assert chunk.get("id"), f"Chunk {i}: missing id"
        assert "created" in chunk, f"Chunk {i}: missing created"
        assert chunk.get("model"), f"Chunk {i}: missing model"
        assert "choices" in chunk, f"Chunk {i}: missing choices"

        # choices can be empty on a usage-only summary chunk
        if not chunk["choices"]:
            continue

        for c_idx, choice in enumerate(chunk["choices"]):
            # Required choice fields in streaming
            assert "index" in choice, f"Chunk {i} choice {c_idx}: missing index"
            assert "delta" in choice, f"Chunk {i} choice {c_idx}: missing delta"
            # Must NOT include full message in streaming
            assert "message" not in choice, f"Chunk {i} choice {c_idx}: message must not appear in streaming"
            # finish_reason must exist; may be null until final chunk
            assert "finish_reason" in choice, f"Chunk {i} choice {c_idx}: missing finish_reason"

            fr = choice.get("finish_reason")
            if fr is not None:
                assert fr in valid_finish_reasons, f"Chunk {i} choice {c_idx}: invalid finish_reason {fr}"
                valid_final_reason_seen = True

    # At least one non-null finish_reason should appear across the stream (finalization)
    assert valid_final_reason_seen, "Expected a final chunk with non-null finish_reason"
