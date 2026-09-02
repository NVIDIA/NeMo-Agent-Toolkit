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
"""Regression tests for components defined in modules using ``from __future__ import annotations``.

With the future import, every annotation in this module is stored as a string and must be resolved against this
module's namespace. ``FunctionInfo``/``FunctionDescriptor`` interpolate the inspected input and output types into
wrappers synthesized inside ``nat.builder.function_info``, so if the string annotations are not resolved against
the defining module first, they are later evaluated against the wrong module globals and fail with ``NameError``
(or produce unresolved Pydantic forward references).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel

from nat.builder.function_info import FunctionDescriptor
from nat.builder.function_info import FunctionInfo


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


async def echo_single(message: EchoInput) -> EchoOutput:
    return EchoOutput(text=message.text)


async def echo_stream(message: EchoInput) -> AsyncGenerator[EchoOutput]:
    yield EchoOutput(text=message.text)


async def concat(first: str, second: EchoInput) -> str:
    return first + second.text


def test_function_descriptor_resolves_future_annotations():
    descriptor = FunctionDescriptor.from_function(echo_single)

    assert descriptor.input_type is EchoInput
    assert descriptor.output_type is EchoOutput
    assert descriptor.input_type_is_base_model
    assert descriptor.output_type_is_base_model


async def test_from_fn_single_fn_resolves_future_annotations():
    info = FunctionInfo.from_fn(echo_single, description="echo")

    assert info.input_type is EchoInput
    assert info.single_output_type is EchoOutput
    assert info.input_schema is EchoInput
    assert info.single_output_schema is EchoOutput

    assert info.single_fn is not None
    assert await info.single_fn(EchoInput(text="hello")) == EchoOutput(text="hello")

    # The auto-synthesized streaming wrapper is defined inside nat.builder.function_info, so it only works if
    # the string annotations were resolved against this module before being interpolated into the wrapper.
    assert info.stream_fn is not None
    chunks = [chunk async for chunk in info.stream_fn(EchoInput(text="hello"))]
    assert chunks == [EchoOutput(text="hello")]


async def test_from_fn_stream_fn_resolves_future_annotations():
    info = FunctionInfo.from_fn(echo_stream, description="echo stream")

    assert info.input_type is EchoInput
    assert info.stream_output_type is EchoOutput
    assert info.stream_output_schema is EchoOutput

    assert info.stream_fn is not None
    chunks = [chunk async for chunk in info.stream_fn(EchoInput(text="hello"))]
    assert chunks == [EchoOutput(text="hello")]


async def test_from_fn_multi_argument_resolves_future_annotations():
    info = FunctionInfo.from_fn(concat, description="concat")

    assert info.input_schema is not None
    assert set(info.input_schema.model_fields) == {"first", "second"}
    assert info.input_schema.model_fields["first"].annotation is str
    assert info.input_schema.model_fields["second"].annotation is EchoInput

    assert info.single_fn is not None
    value = info.input_schema(first="a: ", second=EchoInput(text="b"))
    assert await info.single_fn(value) == "a: b"


async def test_from_fn_zero_argument():
    """Regression test for zero-argument functions (issue #2184)."""

    async def get_status() -> str:
        return "ok"

    info = FunctionInfo.from_fn(get_status, description="get status")

    # The function should be wrapped with an empty input model
    assert info.input_schema is not None
    assert len(info.input_schema.model_fields) == 0

    assert info.single_fn is not None
    # Call with an empty model instance
    value = info.input_schema()
    assert await info.single_fn(value) == "ok"

    # The auto-synthesized streaming wrapper should also work
    assert info.stream_fn is not None
    chunks = [chunk async for chunk in info.stream_fn(info.input_schema())]
    assert chunks == ["ok"]
