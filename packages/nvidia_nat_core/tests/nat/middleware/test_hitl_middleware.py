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
"""Minimal interface tests for HITLMiddleware: prompt presentation and invocation skipping."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.data_models.interactive import BinaryHumanPromptOption
from nat.data_models.interactive import HumanPrompt
from nat.data_models.interactive import HumanPromptBinary
from nat.data_models.interactive import HumanPromptCheckbox
from nat.data_models.interactive import HumanPromptDropdown
from nat.data_models.interactive import HumanPromptNotification
from nat.data_models.interactive import HumanPromptRadio
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import InteractionResponse
from nat.data_models.interactive import MultipleChoiceOption
from nat.middleware.hitl.hitl_middleware import HITLMiddleware
from nat.middleware.hitl.hitl_middleware_config import HITLMiddlewareConfig
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationAction
from nat.middleware.middleware import InvocationContext


class _StubHITLMiddleware(HITLMiddleware):
    """Concrete HITLMiddleware whose abstract decision hooks delegate to swappable mocks."""

    on_pre: AsyncMock
    on_post: AsyncMock

    async def _on_pre_invoke_response(self, response: InteractionResponse,
                                      context: InvocationContext) -> InvocationContext | None:
        return await self.on_pre(response, context)

    async def _on_post_invoke_response(self, response: InteractionResponse,
                                       context: InvocationContext) -> InvocationContext | None:
        return await self.on_post(response, context)


def _mc_option() -> MultipleChoiceOption:
    """A single multiple-choice option for radio/checkbox/dropdown prompts."""
    return MultipleChoiceOption(id="a", label="A", value="a", description="Option A")


# One instance of every public HumanPrompt type, to prove the interface builds and presents each.
_ALL_PROMPTS: list[pytest.ParameterSet] = [
    pytest.param(HumanPromptText(text="t"), id="text"),
    pytest.param(HumanPromptNotification(text="n"), id="notification"),
    pytest.param(
        HumanPromptBinary(
            text="b",
            options=[
                BinaryHumanPromptOption(id="continue", label="Approve", value="continue"),
                BinaryHumanPromptOption(id="cancel", label="Reject", value="cancel"),
            ],
        ),
        id="binary",
    ),
    pytest.param(HumanPromptRadio(text="r", options=[_mc_option()]), id="radio"),
    pytest.param(HumanPromptCheckbox(text="c", options=[_mc_option()]), id="checkbox"),
    pytest.param(HumanPromptDropdown(text="d", options=[_mc_option()]), id="dropdown"),
]

_TEXT_PROMPT: HumanPromptText = HumanPromptText(text="Approve?")


@pytest.fixture(name="function_context")
def fixture_function_context() -> FunctionMiddlewareContext:
    """Static metadata for the wrapped function."""
    return FunctionMiddlewareContext(
        name="test_function",
        config=Mock(),
        description="A test function",
        input_schema=None,
        single_output_schema=type(None),
        stream_output_schema=type(None),
    )


def _make_middleware(*, pre: HumanPrompt | None = None, post: HumanPrompt | None = None) -> _StubHITLMiddleware:
    """Build a stub HITL middleware with the given prompts and no-op (proceed) decision hooks."""
    config: HITLMiddlewareConfig = HITLMiddlewareConfig(pre_invoke_prompt=pre, post_invoke_prompt=post)
    middleware: _StubHITLMiddleware = _StubHITLMiddleware(config=config, builder=Mock())
    middleware.on_pre = AsyncMock(return_value=None)
    middleware.on_post = AsyncMock(return_value=None)
    return middleware


def _invocation_context(function_context: FunctionMiddlewareContext) -> InvocationContext:
    """Build an InvocationContext like the base middleware constructs internally."""
    return InvocationContext(
        function_context=function_context,
        original_args=("input", ),
        original_kwargs={},
        modified_args=("input", ),
        modified_kwargs={},
        output=None,
    )


def _response(text: str) -> InteractionResponse:
    """Wrap a text response in an InteractionResponse envelope."""
    return InteractionResponse(id="1", timestamp="2025-01-01T00:00:00Z", content=HumanResponseText(text=text))


@contextlib.contextmanager
def _patch_user_prompt(response: InteractionResponse) -> Iterator[AsyncMock]:
    """Patch the front-end user-input callback to return the given response."""
    with patch("nat.middleware.hitl.hitl_middleware.Context") as ctx_cls:
        prompt: AsyncMock = AsyncMock(return_value=response)
        ctx_cls.get.return_value.user_interaction_manager.prompt_user_input = prompt
        yield prompt


@pytest.mark.parametrize("phase", ["pre", "post"])
@pytest.mark.parametrize("prompt", _ALL_PROMPTS)
async def test_invoke_phase_with_configured_prompt_presents_prompt_and_forwards_response(
        function_context, phase, prompt):
    """Each public HumanPrompt type is presented verbatim and its response routed to the decision hook."""
    middleware: _StubHITLMiddleware = _make_middleware(**{phase: prompt})
    context: InvocationContext = _invocation_context(function_context)
    response: InteractionResponse = _response("ok")

    method = middleware.pre_invoke if phase == "pre" else middleware.post_invoke
    hook: AsyncMock = middleware.on_pre if phase == "pre" else middleware.on_post

    with _patch_user_prompt(response) as prompt_mock:
        await method(context)

    prompt_mock.assert_awaited_once_with(prompt)
    hook.assert_awaited_once_with(response, context)


@pytest.mark.parametrize("phase", ["pre", "post"])
async def test_invoke_phase_without_prompt_does_not_prompt_user(function_context, phase):
    """With no prompt configured for the phase, the user is never prompted and no decision is made."""
    middleware: _StubHITLMiddleware = _make_middleware()
    context: InvocationContext = _invocation_context(function_context)

    method = middleware.pre_invoke if phase == "pre" else middleware.post_invoke
    hook: AsyncMock = middleware.on_pre if phase == "pre" else middleware.on_post

    with _patch_user_prompt(_response("ok")) as prompt_mock:
        result = await method(context)

    assert result is None
    prompt_mock.assert_not_called()
    hook.assert_not_awaited()


async def test_function_middleware_invoke_when_decision_skips_does_not_invoke_function(function_context):
    """A pre-invoke SKIP decision bypasses the wrapped function and returns None."""
    middleware: _StubHITLMiddleware = _make_middleware(pre=_TEXT_PROMPT)
    call_next: AsyncMock = AsyncMock(return_value="result")

    async def skip(_response, context):
        context.action = InvocationAction.SKIP
        return context

    middleware.on_pre = AsyncMock(side_effect=skip)

    with _patch_user_prompt(_response("no")):
        result = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)

    assert result is None
    call_next.assert_not_called()


async def test_function_middleware_invoke_when_decision_proceeds_invokes_function_and_returns_output(function_context):
    """When the decision proceeds, the wrapped function runs and its output is returned."""
    middleware: _StubHITLMiddleware = _make_middleware(pre=_TEXT_PROMPT)
    call_next: AsyncMock = AsyncMock(return_value="result")

    with _patch_user_prompt(_response("yes")):
        result = await middleware.function_middleware_invoke("input", call_next=call_next, context=function_context)

    assert result == "result"
    call_next.assert_awaited_once()


async def test_function_middleware_stream_when_skip_yields_nothing(function_context):
    """A pre-invoke SKIP decision stops streaming; no chunks are yielded."""
    middleware: _StubHITLMiddleware = _make_middleware(pre=_TEXT_PROMPT)

    async def skip(_resp, ctx):
        ctx.action = InvocationAction.SKIP
        return ctx

    middleware.on_pre = AsyncMock(side_effect=skip)

    async def call_next(*_args, **_kwargs):
        yield "chunk1"
        yield "chunk2"

    with _patch_user_prompt(_response("no")):
        chunks: list[str] = [
            chunk async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                           context=function_context)
        ]

    assert chunks == []


async def test_function_middleware_stream_when_proceed_yields_all_chunks(function_context):
    """When the decision proceeds, all stream chunks are buffered and yielded."""
    middleware: _StubHITLMiddleware = _make_middleware(pre=_TEXT_PROMPT)

    async def call_next(*_args, **_kwargs):
        yield "a"
        yield "b"
        yield "c"

    with _patch_user_prompt(_response("yes")):
        chunks: list[str] = [
            chunk async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                           context=function_context)
        ]

    assert chunks == ["a", "b", "c"]


async def test_function_middleware_stream_post_invoke_clears_output_yields_nothing(function_context):
    """When post-invoke sets context.output to None, that chunk is suppressed and not yielded."""
    middleware: _StubHITLMiddleware = _make_middleware(post=_TEXT_PROMPT)

    async def clear_output(_resp, ctx):
        ctx.output = None
        return ctx

    middleware.on_post = AsyncMock(side_effect=clear_output)

    async def call_next(*_args, **_kwargs):
        yield "x"
        yield "y"

    with _patch_user_prompt(_response("no")):
        chunks: list[str] = [
            chunk async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                           context=function_context)
        ]

    assert chunks == []


async def test_function_middleware_stream_post_invoke_called_per_chunk(function_context):
    """post_invoke is called once per chunk with the individual chunk value."""
    middleware: _StubHITLMiddleware = _make_middleware(post=_TEXT_PROMPT)
    received_outputs: list[Any] = []

    async def capture(_resp, ctx):
        received_outputs.append(ctx.output)

    middleware.on_post = AsyncMock(side_effect=capture)

    async def call_next(*_args, **_kwargs):
        yield "p"
        yield "q"

    with _patch_user_prompt(_response("ok")):
        chunks: list[str] = [
            chunk async for chunk in middleware.function_middleware_stream("input", call_next=call_next,
                                                                           context=function_context)
        ]

    assert received_outputs == ["p", "q"]
    assert chunks == ["p", "q"]
