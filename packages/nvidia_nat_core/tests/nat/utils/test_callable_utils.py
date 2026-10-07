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

import pytest

from nat.utils.callable_utils import ainvoke_any
from nat.utils.callable_utils import is_async_callable


async def test_ainvoke_any_sync_function():

    def func(value):
        return value * 2

    assert await ainvoke_any(func, 3) == 6


async def test_ainvoke_any_async_function():

    async def func(value):
        return value * 2

    assert await ainvoke_any(func, 3) == 6


async def test_ainvoke_any_generator_return_value_wins():
    """An explicit return value is the result, as it always has been."""

    def func():
        yield 1
        yield 2
        return "returned"

    assert await ainvoke_any(func) == "returned"


async def test_ainvoke_any_generators_agree_on_the_last_yielded_value():
    """A yield-only generator reports its last value whether it is sync or async.

    PEP 525 forbids `return <value>` in an async generator, so the last yielded
    value is all one can report. The sync branch returned only `StopIteration.value`,
    which is None for a generator that just yields, so the two disagreed: `sync`
    below evaluated to None while `asynchronous` evaluated to 2.
    """

    def sync():
        yield 1
        yield 2

    async def asynchronous():
        yield 1
        yield 2

    assert await ainvoke_any(sync) == 2
    assert await ainvoke_any(asynchronous) == 2


async def test_ainvoke_any_empty_generators_return_none():

    def sync():
        return
        yield

    async def asynchronous():
        return
        yield

    assert await ainvoke_any(sync) is None
    assert await ainvoke_any(asynchronous) is None


async def test_ainvoke_any_generator_is_consumed_to_exhaustion():
    """The side effects of every step happen, not just the first."""
    seen = []

    def func():
        for i in range(3):
            seen.append(i)
            yield i

    assert await ainvoke_any(func) == 2
    assert seen == [0, 1, 2]


@pytest.mark.parametrize("truthy", [True, "redact", 1])
async def test_ainvoke_any_truthiness_survives_a_sync_generator(truthy):
    """The shape `RedactionProcessor` relies on: `bool(await ainvoke_any(...))`.

    A sync generator yielding a truthy value used to come back as None, so a
    redaction callback written that way decided not to redact.
    """

    def func():
        yield truthy

    assert bool(await ainvoke_any(func)) is True


def test_is_async_callable():

    def sync():
        pass

    async def coro():
        pass

    async def agen():
        yield 1

    assert is_async_callable(coro)
    assert is_async_callable(agen)
    assert not is_async_callable(sync)
