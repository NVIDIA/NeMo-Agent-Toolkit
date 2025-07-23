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

import logging

import pytest

from aiq.experimental.decorators.experimental_warning_decorator import WARNING_MESSAGE
from aiq.experimental.decorators.experimental_warning_decorator import aiq_experimental
from aiq.experimental.decorators.experimental_warning_decorator import issue_warning
from aiq.experimental.decorators.experimental_warning_decorator import warning_issued


# Reset warning state before each test
@pytest.fixture(autouse=True)
def clear_warnings():
    warning_issued.clear()
    yield
    warning_issued.clear()


def test_sync_function_logs_warning_once(caplog):
    caplog.set_level(logging.WARNING)

    @aiq_experimental
    def foo(x):
        return x + 1

    # first call should log
    assert foo(1) == 2
    assert any(WARNING_MESSAGE in rec.message for rec in caplog.records)

    caplog.clear()

    # second call should not log again
    assert foo(2) == 3
    assert not caplog.records


async def test_async_function_logs_warning_once(caplog):
    caplog.set_level(logging.WARNING)

    @aiq_experimental
    async def bar(x):
        return x * 2

    # first await should log
    result1 = await bar(3)
    assert result1 == 6
    assert any(WARNING_MESSAGE in rec.message for rec in caplog.records)

    caplog.clear()

    # second await should not log again
    result2 = await bar(4)
    assert result2 == 8
    assert not caplog.records


def test_sync_generator_logs_and_yields(caplog):
    caplog.set_level(logging.WARNING)

    @aiq_experimental
    def gen(n):
        for i in range(n):
            yield i

    # iterate first time
    out = list(gen(3))
    assert out == [0, 1, 2]
    assert any(WARNING_MESSAGE in rec.message for rec in caplog.records)

    caplog.clear()

    # iterate second time: still only one warning ever
    out2 = list(gen(2))
    assert out2 == [0, 1]
    assert not caplog.records


async def test_async_generator_logs_and_yields(caplog):
    caplog.set_level(logging.WARNING)

    @aiq_experimental
    async def agen(n):
        for i in range(n):
            yield i

    # async iteration via __anext__
    collected = []
    async for v in agen(4):
        collected.append(v)
    assert collected == [0, 1, 2, 3]
    assert any(WARNING_MESSAGE in rec.message for rec in caplog.records)

    caplog.clear()

    # second iteration no new warning
    collected2 = []
    async for v in agen(2):
        collected2.append(v)
    assert collected2 == [0, 1]
    assert not caplog.records


def test_issue_warning_idempotent(caplog):
    caplog.set_level(logging.WARNING)

    # directly issue warning twice
    issue_warning("myfunc")
    issue_warning("myfunc")

    records = [r for r in caplog.records if WARNING_MESSAGE in r.message]
    assert len(records) == 1


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):

        @aiq_experimental(metadata="not-a-dict")
        def f1():
            pass


def test_metadata_keys_must_be_str():
    with pytest.raises(TypeError):

        @aiq_experimental(metadata={1: "value"})
        def f2():
            pass
