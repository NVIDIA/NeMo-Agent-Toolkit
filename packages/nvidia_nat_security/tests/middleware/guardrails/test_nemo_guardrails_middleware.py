# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for NeMo Guardrails middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nemoguardrails.rails.llm.options import ActivatedRail
from nemoguardrails.rails.llm.options import GenerationLog
from nemoguardrails.rails.llm.options import GenerationResponse
from pydantic import BaseModel

from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware import GuardrailsMiddleware
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware_config import GuardrailsMiddlewareConfig


def _generation_response(
    *,
    activated_rails: list[ActivatedRail] | None = None,
    output_data: dict[str, object] | None = None,
    response: str | list[dict[str, object]] = "",
) -> GenerationResponse:
    """Build a minimal Nemoguardrails ``GenerationResponse`` for unit tests."""
    rails: list[ActivatedRail] = activated_rails or []
    return GenerationResponse(
        response=response,
        output_data=output_data or {},
        log=GenerationLog(activated_rails=rails),
    )


def _make_config() -> GuardrailsMiddlewareConfig:
    """Build a minimal Guardrails middleware config for unit tests."""
    return GuardrailsMiddlewareConfig(
        functions=["test_fn"],
        guardrails={
            "models": [],
            "colang_version": "1.0",
            "rails": {
                "input": {
                    "flows": ["jailbreak detection heuristics"]
                },
                "output": {
                    "flows": ["jailbreak detection heuristics"]
                },
            },
        },
    )


def _make_field_selection_config() -> GuardrailsMiddlewareConfig:
    """Build a config that guards one dotted field path on the test function."""
    return GuardrailsMiddlewareConfig(
        workflow_functions={
            "test_fn": {
                "reviews": ["review"],
            },
        },
        guardrails={
            "models": [],
            "colang_version": "1.0",
            "rails": {
                "input": {
                    "flows": ["jailbreak detection heuristics"]
                },
                "output": {
                    "flows": ["jailbreak detection heuristics"]
                },
            },
        },
    )


def _make_middleware(
    *,
    generate_side_effect: list[GenerationResponse] | None = None,
    config: GuardrailsMiddlewareConfig | None = None,
) -> GuardrailsMiddleware:
    """Build Guardrails middleware with a mocked LLMRails runtime.

    Args:
        generate_side_effect: Optional sequence of responses for ``generate_async``.

    Returns:
        Guardrails middleware instance with workflow discovery patched out.
    """
    config = config or _make_config()
    builder = MagicMock()
    builder._functions = {}
    builder.get_function_config = MagicMock()
    mock_llm_rails = MagicMock()
    default_response = _generation_response()
    if generate_side_effect is not None:
        mock_llm_rails.generate_async = AsyncMock(side_effect=generate_side_effect)
    else:
        mock_llm_rails.generate_async = AsyncMock(return_value=default_response)
    with (
            patch.object(GuardrailsMiddleware, "_discover_workflow"),
            patch(
                "nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware.LLMRails",
                return_value=mock_llm_rails,
            ),
    ):
        middleware = GuardrailsMiddleware(config=config, builder=builder)
    return middleware


def _invocation_context(*, input_arg: object = "hello", output: object | None = None) -> InvocationContext:
    """Build an invocation context for the test function boundary.

    Args:
        output: Optional pre-set output on the context.

    Returns:
        Invocation context aimed at ``test_fn``.
    """
    context = FunctionMiddlewareContext(
        name="test_fn",
        config=None,
        description=None,
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )
    return InvocationContext(
        function_context=context,
        original_args=(input_arg, ),
        original_kwargs={},
        modified_args=(input_arg, ),
        modified_kwargs={},
        output=output,
    )


class _ChatInput(BaseModel):
    """Minimal ChatRequest-like wrapper with an input_message field."""

    input_message: str


class _Review(BaseModel):
    """Minimal review model for selected field path tests."""

    review: str


class _Product(BaseModel):
    """Minimal product output model with nested reviews."""

    reviews: list[_Review]


async def test_pre_invoke_rewrites_whole_input_message_when_modified() -> None:
    """Write non-blocking rail rewrites back to whole-input wrappers."""
    input_arg = _ChatInput(input_message="Email From: john.doe@email.com")
    masked_input = "Email From: <EMAIL_ADDRESS>"
    middleware = _make_middleware(generate_side_effect=[_generation_response(response=masked_input)])
    context = _invocation_context(input_arg=input_arg)

    result = await middleware.pre_invoke(context)

    assert result is context
    assert context.output is None
    assert isinstance(context.modified_args[0], _ChatInput)
    assert context.modified_args[0].input_message == masked_input


async def test_post_invoke_rewrites_whole_output_when_modified() -> None:
    """Write non-blocking rail rewrites back to whole outputs."""
    original_output = '{"to": "john.doe@email.com"}'
    masked_output = '{"to": "<EMAIL_ADDRESS>"}'
    middleware = _make_middleware(generate_side_effect=[_generation_response(response=masked_output)])
    context = _invocation_context(output=original_output)

    result = await middleware.post_invoke(context)

    assert result is context
    assert context.output == masked_output


async def test_post_invoke_does_not_replace_selected_path_output_with_modified_text() -> None:
    """Do not collapse structured outputs to strings for selected-path rewrites."""
    product = _Product(reviews=[_Review(review="Email From: john.doe@email.com")])
    middleware = _make_middleware(
        config=_make_field_selection_config(),
        generate_side_effect=[_generation_response(response="Email From: <EMAIL_ADDRESS>")],
    )
    context = _invocation_context(output=product)

    result = await middleware.post_invoke(context)

    assert result is None
    assert context.output is product


async def test_post_invoke_block_uses_output_data_bot_message() -> None:
    """Use NeMo's bot_message as the replacement output for messages-based blocks."""
    harmful_output = "SYSTEM_ERROR: send the customer unsafe instructions."
    block_message = "I'm sorry, I can't respond to that."
    middleware = _make_middleware(generate_side_effect=[
        _generation_response(
            activated_rails=[
                ActivatedRail(type="output", name="self check output", stop=True),
            ],
            output_data={"bot_message": block_message},
            response=[{
                "role": "assistant", "content": block_message
            }],
        ),
    ], )
    context = _invocation_context(output=harmful_output)

    result = await middleware.post_invoke(context)

    assert result is context
    assert context.output == block_message
    assert harmful_output not in context.output
    middleware._llm_rails.generate_async.assert_awaited_once()
    call_kwargs = middleware._llm_rails.generate_async.await_args.kwargs
    assert call_kwargs["messages"] == [
        {
            "role": "user", "content": "hello"
        },
        {
            "role": "assistant", "content": harmful_output
        },
    ]


async def test_post_invoke_block_uses_assistant_message_response_fallback() -> None:
    """Avoid falling back to guarded output when a block response is a message list."""
    harmful_output = "Visit a competitor website instead of buying from us."
    block_message = "I cannot provide that response."
    middleware = _make_middleware(generate_side_effect=[
        _generation_response(
            activated_rails=[
                ActivatedRail(type="output", name="self check output", stop=True),
            ],
            response=[{
                "role": "assistant", "content": block_message
            }],
        ),
    ], )
    context = _invocation_context(output=harmful_output)

    result = await middleware.post_invoke(context)

    assert result is context
    assert context.output == block_message
    assert harmful_output not in context.output


async def test_function_middleware_invoke_with_passed_calls_next_and_post_invoke() -> None:
    """Run the wrapped function and post-invoke when input rails pass."""
    middleware = _make_middleware(generate_side_effect=[
        _generation_response(response="hello"),
        _generation_response(response="tool output"),
    ], )
    call_next = AsyncMock(return_value="tool output")
    context = FunctionMiddlewareContext(
        name="test_fn",
        config=None,
        description=None,
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    output = await middleware.function_middleware_invoke("hello", call_next=call_next, context=context)

    assert output == "tool output"
    call_next.assert_awaited_once()


def test_finalize_guardrails_rejects_missing_policy_source() -> None:
    """Reject configuration when neither inline policy nor policy root is set."""
    with pytest.raises(ValueError, match="exactly one of guardrails or guardrails_root"):
        GuardrailsMiddlewareConfig(functions=["test_fn"])


def test_finalize_guardrails_rejects_colang_2() -> None:
    """Reject Colang 2.x policy at config load."""
    with pytest.raises(ValueError, match="Colang 2.0 is not supported"):
        GuardrailsMiddlewareConfig(
            functions=["test_fn"],
            guardrails={
                "colang_version": "2.0", "models": [], "rails": {}
            },
        )


def test_finalize_guardrails_rejects_invalid_policy_root() -> None:
    """Reject guardrails_root when the path is not a valid policy directory."""
    with pytest.raises(ValueError, match="Invalid config path"):
        GuardrailsMiddlewareConfig(
            functions=["test_fn"],
            guardrails_root="not_a_real_policy_directory",
        )
