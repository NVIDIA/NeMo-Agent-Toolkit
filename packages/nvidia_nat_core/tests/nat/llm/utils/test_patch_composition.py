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
"""The framework LLM wrappers apply patch_with_retry and then patch_with_thinking; both must stay in effect."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest

from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


class RateLimited(Exception):
    status_code = 429


class Client:
    """Fails its first call with a 429 and records the messages of every attempt."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def _attempt(self, messages: list[str]):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            raise RateLimited("429 Too Many Requests")

    def invoke(self, messages: list[str]) -> str:
        self._attempt(messages)
        return "ok"

    async def ainvoke(self, messages: list[str]) -> str:
        self._attempt(messages)
        return "ok"

    async def astream(self, messages: list[str]) -> AsyncGenerator[str, None]:
        self._attempt(messages)
        yield "ok"


class PrependThink(BaseThinkingInjector):

    def inject(self, messages: list[str], *args: Any, **kwargs: Any) -> FunctionArgumentWrapper:
        return FunctionArgumentWrapper(["/think", *messages], *args, **kwargs)


def _retry(client):
    return patch_with_retry(client, retries=3, retry_codes=[429])


def _think(client):
    return patch_with_thinking(client,
                               PrependThink(system_prompt="/think", function_names=["invoke", "ainvoke", "astream"]))


ORDERS = {"retry_then_thinking": (_retry, _think), "thinking_then_retry": (_think, _retry)}


@pytest.fixture(name="no_sleep", autouse=True)
def fixture_no_sleep():
    with patch("asyncio.sleep"), patch("time.sleep"):
        yield


@pytest.mark.parametrize("order", ORDERS, ids=list(ORDERS))
async def test_async_method_keeps_retry_and_thinking(order):
    client = Client()
    for apply in ORDERS[order]:
        client = apply(client)

    assert await client.ainvoke(["hi"]) == "ok"
    assert client.calls == [["/think", "hi"], ["/think", "hi"]]


@pytest.mark.parametrize("order", ORDERS, ids=list(ORDERS))
async def test_async_generator_method_keeps_retry_and_thinking(order):
    client = Client()
    for apply in ORDERS[order]:
        client = apply(client)

    assert [chunk async for chunk in client.astream(["hi"])] == ["ok"]
    assert client.calls == [["/think", "hi"], ["/think", "hi"]]


@pytest.mark.parametrize("order", ORDERS, ids=list(ORDERS))
def test_sync_method_keeps_retry_and_thinking(order):
    client = Client()
    for apply in ORDERS[order]:
        client = apply(client)

    assert client.invoke(["hi"]) == "ok"
    assert client.calls == [["/think", "hi"], ["/think", "hi"]]


def test_patching_one_instance_leaves_other_instances_alone():
    patched = _think(_retry(Client()))
    plain = Client()

    assert patched.invoke(["hi"]) == "ok"
    with pytest.raises(RateLimited):
        plain.invoke(["hi"])
    assert plain.calls == [["hi"]]
