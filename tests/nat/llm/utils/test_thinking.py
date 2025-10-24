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

from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking


class MockClass:

    def sync_method(self, *args, **kwargs):
        return (args, kwargs)

    async def async_method(self, *args, **kwargs):
        return (args, kwargs)

    def gen_method(self, *args, **kwargs):
        yield (args, kwargs)

    async def agen_method(self, *args, **kwargs):
        yield (args, kwargs)


class AddThinking(BaseThinkingInjector):

    def inject(self, x: str, *args, **kwargs) -> FunctionArgumentWrapper:
        return FunctionArgumentWrapper(("thinking " + x), *args, **kwargs)


class AddThinkingWithArgs(BaseThinkingInjector):

    def inject(self, *args, **kwargs) -> FunctionArgumentWrapper:
        return FunctionArgumentWrapper("thinking", *args, **kwargs)


class AddThinkingWithKwargs(BaseThinkingInjector):

    def inject(self, *args, **kwargs) -> FunctionArgumentWrapper:
        return FunctionArgumentWrapper(*args, thinking=True, **kwargs)


@pytest.mark.asyncio
async def test_patch_with_thinking_in_place():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(
        mock_obj,
        AddThinking(
            system_prompt="thinking",
            function_names=[
                "sync_method",
                "async_method",
                "gen_method",
                "agen_method",
            ],
        ),
    )
    assert patched_obj is mock_obj

    expected = (("thinking test", *args), kwargs)

    actual = patched_obj.sync_method("test", *args, **kwargs)
    assert actual == expected

    actual = await patched_obj.async_method("test", *args, **kwargs)
    assert actual == expected

    for item in patched_obj.gen_method("test", *args, **kwargs):
        assert item == expected

    async for item in patched_obj.agen_method("test", *args, **kwargs):
        assert item == expected


@pytest.mark.asyncio
async def test_patch_with_thinking_modify_args():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(
        mock_obj,
        AddThinkingWithArgs(
            system_prompt="thinking",
            function_names=[
                "sync_method",
                "async_method",
                "gen_method",
                "agen_method",
            ],
        ),
    )
    assert patched_obj is mock_obj

    expected = (("thinking", "test", *args), kwargs)

    actual = patched_obj.sync_method("test", *args, **kwargs)
    assert actual == expected

    actual = await patched_obj.async_method("test", *args, **kwargs)
    assert actual == expected

    for item in patched_obj.gen_method("test", *args, **kwargs):
        assert item == expected

    async for item in patched_obj.agen_method("test", *args, **kwargs):
        assert item == expected


@pytest.mark.asyncio
async def test_patch_with_thinking_modify_kwargs():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(
        mock_obj,
        AddThinkingWithKwargs(
            system_prompt="thinking",
            function_names=[
                "sync_method",
                "async_method",
                "gen_method",
                "agen_method",
            ],
        ),
    )
    assert patched_obj is mock_obj

    expected = (("test", *args), {"thinking": True, **kwargs})

    actual = patched_obj.sync_method("test", *args, **kwargs)
    assert actual == expected

    actual = await patched_obj.async_method("test", *args, **kwargs)
    assert actual == expected

    for item in patched_obj.gen_method("test", *args, **kwargs):
        assert item == expected

    async for item in patched_obj.agen_method("test", *args, **kwargs):
        assert item == expected
