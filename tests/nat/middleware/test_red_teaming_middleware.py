# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the RedTeamingMiddleware functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nat.middleware.function_middleware import FunctionMiddlewareContext
from nat.middleware.red_teaming.red_teaming_middleware import RedTeamingMiddleware


class UserInfo(BaseModel):
    name: str
    email: str


class RequestData(BaseModel):
    query: str
    context: str


class LLMInput(BaseModel):
    prompt: str
    system_message: str
    temperature: float


class LLMOutput(BaseModel):
    response: str
    confidence: float


class NestedInput(BaseModel):
    user: UserInfo
    request: RequestData


class NestedOutput(BaseModel):
    result: str
    metadata: dict


class MultiFieldModel(BaseModel):
    messages: list[str]


async def test_simple_output_replace_strategy():
    """Test simple string input/output with replace strategy on output."""
    middleware = RedTeamingMiddleware(
        attack_payload="REPLACED",
        payload_placement="replace",
        target_location="output",
    )

    mock_call_next = AsyncMock(return_value="original output")

    context = FunctionMiddlewareContext(
        name="simple_function",
        config=MagicMock(),
        description="Simple function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke("hello", call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once_with("hello")
    assert result == "REPLACED"


@pytest.mark.parametrize(
    "call_limit,expected_results",
    [
        (None, ["REPLACED", "REPLACED", "REPLACED"]),
        (1, ["REPLACED", "second output", "third output"]),
        (2, ["REPLACED", "REPLACED", "third output"]),
    ],
)
async def test_call_limit(call_limit, expected_results):
    """Test that call_limit controls how many times the payload is applied."""
    middleware = RedTeamingMiddleware(
        attack_payload="REPLACED",
        payload_placement="replace",
        target_location="output",
        call_limit=call_limit,
    )

    context = FunctionMiddlewareContext(
        name="simple_function",
        config=MagicMock(),
        description="Simple function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    outputs = ["first output", "second output", "third output"]
    results = []
    for i, output in enumerate(outputs):
        mock_call_next = AsyncMock(return_value=output)
        result = await middleware.function_middleware_invoke(f"input{i}", call_next=mock_call_next, context=context)
        results.append(result)

    assert results == expected_results


async def test_attack_nested_input_field():
    """Attack a nested field in input via function_middleware_invoke."""
    middleware = RedTeamingMiddleware(
        attack_payload="INJECTED",
        target_field="$.user.email",
        payload_placement="replace",
        target_location="input",
    )

    input_value = NestedInput(
        user=UserInfo(name="Alice", email="alice@example.com"),
        request=RequestData(query="What is AI?", context="Tech support"),
    )

    mock_call_next = AsyncMock(return_value=NestedOutput(result="Answer", metadata={}))

    context = FunctionMiddlewareContext(
        name="test_function",
        config=MagicMock(),
        description="Test",
        input_schema=NestedInput,
        single_output_schema=NestedOutput,
        stream_output_schema=None,
    )

    await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once()
    received_input = mock_call_next.call_args.args[0]
    assert received_input.user.email == "INJECTED"
    assert received_input.user.name == "Alice"
    assert received_input.request.query == "What is AI?"


async def test_attack_input_with_output_passthrough():
    """Verify output is unchanged when attacking input."""
    middleware = RedTeamingMiddleware(
        attack_payload="PAYLOAD ",
        target_field="$.prompt",
        payload_placement="append_start",
        target_location="input",
    )

    input_value = LLMInput(prompt="Hello world", system_message="Be helpful", temperature=0.7)
    expected_output = LLMOutput(response="Hi there!", confidence=0.95)

    mock_call_next = AsyncMock(return_value=expected_output)

    context = FunctionMiddlewareContext(
        name="llm.generate",
        config=MagicMock(),
        description="Generate",
        input_schema=LLMInput,
        single_output_schema=LLMOutput,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once()
    assert result.response == "Hi there!"
    assert result.confidence == 0.95


async def test_attack_deeply_nested_jsonpath():
    """Attack a deeply nested field using jsonpath."""
    middleware = RedTeamingMiddleware(
        attack_payload=" [CONTEXT INJECTED]",
        target_field="$.request.context",
        payload_placement="append_end",
        target_location="input",
    )

    input_value = NestedInput(
        user=UserInfo(name="Bob", email="bob@test.com"),
        request=RequestData(query="Help me", context="Customer service"),
    )

    mock_call_next = AsyncMock(return_value=NestedOutput(result="Done", metadata={"status": "ok"}))

    context = FunctionMiddlewareContext(
        name="service.handle",
        config=MagicMock(),
        description="Handle request",
        input_schema=NestedInput,
        single_output_schema=NestedOutput,
        stream_output_schema=None,
    )

    await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once()
    received_input = mock_call_next.call_args.args[0]
    assert received_input.request.context == "Customer service [CONTEXT INJECTED]"
    assert received_input.request.query == "Help me"


async def test_attack_nested_output_field():
    """Attack a field in the output via function_middleware_invoke."""
    middleware = RedTeamingMiddleware(
        attack_payload="MALICIOUS RESPONSE",
        target_field="$.response",
        payload_placement="replace",
        target_location="output",
    )

    input_value = LLMInput(prompt="Hello", system_message="Be nice", temperature=0.5)
    mock_call_next = AsyncMock(return_value=LLMOutput(response="Original response", confidence=0.9))

    context = FunctionMiddlewareContext(
        name="llm.chat",
        config=MagicMock(),
        description="Chat",
        input_schema=LLMInput,
        single_output_schema=LLMOutput,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once()
    assert result.response == "MALICIOUS RESPONSE"
    assert result.confidence == 0.9


async def test_attack_output_preserves_input():
    """Verify input is passed unchanged when attacking output."""
    middleware = RedTeamingMiddleware(
        attack_payload=" APPENDED",
        target_field="$.result",
        payload_placement="append_end",
        target_location="output",
    )

    input_value = NestedInput(
        user=UserInfo(name="Carol", email="carol@test.com"),
        request=RequestData(query="Question", context="Context"),
    )

    mock_call_next = AsyncMock(return_value=NestedOutput(result="Success", metadata={"key": "value"}))

    context = FunctionMiddlewareContext(
        name="processor.run",
        config=MagicMock(),
        description="Process",
        input_schema=NestedInput,
        single_output_schema=NestedOutput,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    # Input should be unchanged
    mock_call_next.assert_called_once()
    received_input = mock_call_next.call_args.args[0]
    assert received_input.user.name == "Carol"
    assert received_input.user.email == "carol@test.com"
    # Output should be modified
    assert result.result == "Success APPENDED"


async def test_target_function_filtering():
    """Middleware skips non-targeted functions."""
    middleware = RedTeamingMiddleware(
        attack_payload="ATTACK",
        target_field="$.prompt",
        payload_placement="replace",
        target_location="input",
        target_function_or_group="other_function",
    )

    input_value = LLMInput(prompt="Original", system_message="System", temperature=0.5)
    mock_call_next = AsyncMock(return_value=LLMOutput(response="Response", confidence=0.8))

    context = FunctionMiddlewareContext(
        name="llm.generate",
        config=MagicMock(),
        description="Generate",
        input_schema=LLMInput,
        single_output_schema=LLMOutput,
        stream_output_schema=None,
    )

    await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    # Input should NOT be modified since function is not targeted
    mock_call_next.assert_called_once()
    received_input = mock_call_next.call_args.args[0]
    assert received_input.prompt == "Original"


async def test_multiple_field_matches_with_all_strategy():
    """Test resolution strategy 'all' modifies all matching fields."""
    middleware = RedTeamingMiddleware(
        attack_payload="INJECTED",
        target_field="$.messages[*]",
        payload_placement="replace",
        target_location="input",
        target_field_resolution_strategy="all",
    )

    input_value = MultiFieldModel(messages=["first", "second", "third"])
    mock_call_next = AsyncMock(return_value={"status": "ok"})

    context = FunctionMiddlewareContext(
        name="processor.batch",
        config=MagicMock(),
        description="Batch process",
        input_schema=MultiFieldModel,
        single_output_schema=dict,
        stream_output_schema=None,
    )

    await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once()
    received_input = mock_call_next.call_args.args[0]
    assert received_input.messages == ["INJECTED", "INJECTED", "INJECTED"]


async def test_multiple_field_matches_with_first_strategy():
    """Test resolution strategy 'first' modifies only the first match."""
    middleware = RedTeamingMiddleware(
        attack_payload="INJECTED",
        target_field="$.messages[*]",
        payload_placement="replace",
        target_location="input",
        target_field_resolution_strategy="first",
    )

    input_value = MultiFieldModel(messages=["first", "second", "third"])
    mock_call_next = AsyncMock(return_value={"status": "ok"})

    context = FunctionMiddlewareContext(
        name="processor.batch",
        config=MagicMock(),
        description="Batch process",
        input_schema=MultiFieldModel,
        single_output_schema=dict,
        stream_output_schema=None,
    )

    await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)

    mock_call_next.assert_called_once()
    received_input = mock_call_next.call_args.args[0]
    assert received_input.messages == ["INJECTED", "second", "third"]


async def test_multiple_field_matches_with_error_strategy():
    """Test resolution strategy 'error' raises ValueError on multiple matches."""
    middleware = RedTeamingMiddleware(
        attack_payload="INJECTED",
        target_field="$.messages[*]",
        payload_placement="replace",
        target_location="input",
        target_field_resolution_strategy="error",
    )

    input_value = MultiFieldModel(messages=["first", "second", "third"])
    mock_call_next = AsyncMock(return_value={"status": "ok"})

    context = FunctionMiddlewareContext(
        name="processor.batch",
        config=MagicMock(),
        description="Batch process",
        input_schema=MultiFieldModel,
        single_output_schema=dict,
        stream_output_schema=None,
    )

    with pytest.raises(ValueError, match="Multiple matches found"):
        await middleware.function_middleware_invoke(input_value, call_next=mock_call_next, context=context)


@pytest.mark.parametrize(
    "placement,original,expected",
    [
        ("replace", "original text", "PAYLOAD"),
        ("append_start", "original text", "PAYLOADoriginal text"),
        ("append_end", "original text", "original textPAYLOAD"),
        ("append_middle", "First sentence. Second sentence.", "First sentence. PAYLOADSecond sentence."),
    ],
)
def test_string_placement_modes(placement, original, expected):
    """Test all payload placement modes for string values."""
    middleware = RedTeamingMiddleware(attack_payload="PAYLOAD", payload_placement=placement)
    result = middleware._apply_payload_to_function_value(original)
    assert result == expected
