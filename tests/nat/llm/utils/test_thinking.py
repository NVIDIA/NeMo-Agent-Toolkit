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

from nat.llm.utils.thinking import patch_with_thinking


class MockClass:

    def sync_method(self, message: str, *args, **kwargs):
        return (message, args, kwargs)

    async def async_method(self, message: str, *args, **kwargs):
        return (message, args, kwargs)

    def gen_method(self, message: str, *args, **kwargs):
        yield (message, args, kwargs)

    async def agen_method(self, message: str, *args, **kwargs):
        yield (message, args, kwargs)


def add_thinking(x: str) -> str:
    return "thinking " + x


def test_patch_with_thinking_sync():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(mock_obj, ["sync_method"], add_thinking)
    assert patched_obj is mock_obj
    actual = patched_obj.sync_method("test", *args, **kwargs)
    assert actual == ("thinking test", args, kwargs)


@pytest.mark.asyncio
async def test_patch_with_thinking_async():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(mock_obj, ["async_method"], add_thinking)
    assert patched_obj is mock_obj
    actual = await patched_obj.async_method("test", *args, **kwargs)
    assert actual == ("thinking test", args, kwargs)


def test_patch_with_thinking_gen():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(mock_obj, ["gen_method"], add_thinking)
    assert patched_obj is mock_obj
    for item in patched_obj.gen_method("test", *args, **kwargs):
        assert item == ("thinking test", args, kwargs)


@pytest.mark.asyncio
async def test_patch_with_thinking_agen():
    args = (
        123,
        "foo",
        None,
    )
    kwargs = {"foo": "bar", "baz": 123}
    mock_obj = MockClass()
    patched_obj = patch_with_thinking(mock_obj, ["agen_method"], add_thinking)
    assert patched_obj is mock_obj
    async for item in patched_obj.agen_method("test", *args, **kwargs):
        assert item == ("thinking test", args, kwargs)
