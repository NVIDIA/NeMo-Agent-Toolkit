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
"""Automatic retries wait for the time a rate-limited provider asks for in `Retry-After`."""

import sys
from collections.abc import AsyncGenerator
from collections.abc import Generator
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nat.utils.exception_handlers import automatic_retries
from nat.utils.exception_handlers.automatic_retries import _retry_after_seconds
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


class RateLimited(Exception):
    """Like LiteLLM's and OpenAI's RateLimitError: a status code and the HTTP response."""

    status_code = 429

    def __init__(self, headers: dict[str, str] | None = None):
        super().__init__("429 Too Many Requests")
        self.response = SimpleNamespace(headers=headers or {})


class Throttled(Exception):
    """Like botocore's ClientError for Bedrock throttling."""

    def __init__(self, headers: dict[str, str]):
        super().__init__("ThrottlingException: 429 Too Many Requests")
        self.response = {"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {"HTTPHeaders": headers}}


@pytest.mark.parametrize("exc, expected",
                         [
                             (RateLimited({"Retry-After": "3"}), 3.0),
                             (RateLimited({"retry-after": "1.5"}), 1.5),
                             (RateLimited({"retry-after-ms": "1500"}), 1.5),
                             (Throttled({"retry-after": "2"}), 2.0),
                             (RateLimited({"Retry-After": "soon"}), None),
                             (RateLimited({"Retry-After": "-5"}), None),
                             (RateLimited({
                                 "retry-after-ms": "-1", "Retry-After": "30"
                             }), 30.0),
                             (RateLimited({"Retry-After": "inf"}), None),
                             (RateLimited({
                                 "retry-after-ms": "Infinity", "Retry-After": "30"
                             }), 30.0),
                             (RateLimited({}), None),
                             (Exception("no response at all"), None),
                         ],
                         ids=[
                             "seconds",
                             "lowercase_fractional",
                             "milliseconds",
                             "botocore",
                             "garbage",
                             "negative",
                             "negative_ms_falls_back",
                             "infinite",
                             "infinite_ms_falls_back",
                             "missing",
                             "plain"
                         ])
def test_retry_after_seconds(exc, expected):
    assert _retry_after_seconds(exc) == expected


def test_retry_after_as_an_http_date():
    when = datetime.now(UTC) + timedelta(seconds=10)
    wait = _retry_after_seconds(RateLimited({"Retry-After": format_datetime(when, usegmt=True)}))
    assert wait is not None and 8 <= wait <= 10


def test_retry_after_date_in_the_past_waits_nothing():
    when = datetime.now(UTC) - timedelta(seconds=30)
    assert _retry_after_seconds(RateLimited({"Retry-After": format_datetime(when, usegmt=True)})) == 0.0


class Client:
    """Fails its first `failures` calls with `error`, then succeeds."""

    def __init__(self, error: Exception, failures: int = 1):
        self.error = error
        self.failures = failures
        self.calls = 0

    def _attempt(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error

    def invoke(self) -> str:
        self._attempt()
        return "ok"

    async def ainvoke(self) -> str:
        self._attempt()
        return "ok"

    def stream(self) -> Generator[str]:
        self._attempt()
        yield "ok"

    async def astream(self) -> AsyncGenerator[str]:
        self._attempt()
        yield "ok"


async def _call(client: Client, method: str):
    if method == "invoke":
        return client.invoke()
    if method == "ainvoke":
        return await client.ainvoke()
    if method == "stream":
        return list(client.stream())
    return [chunk async for chunk in client.astream()]


@pytest.fixture(name="sleeps")
def sleeps_fixture():
    """Record the waits requested by the retry wrapper.

    `asyncio.sleep` and `time.sleep` are patched globally, so anything else that sleeps while a test runs (a thread
    or task left behind by an earlier test, library internals) would also be recorded and make the assertions
    order-dependent. Only calls made from `automatic_retries` are kept; everything else just returns immediately.
    """
    waits: list[float] = []

    def _called_from_retry_module() -> bool:
        # frame 0 is this helper, 1 is the fake sleep, 2 is whoever called sleep
        return sys._getframe(2).f_globals.get("__name__") == automatic_retries.__name__

    async def fake_async_sleep(seconds):
        if _called_from_retry_module():
            waits.append(seconds)

    def fake_sleep(seconds):
        if _called_from_retry_module():
            waits.append(seconds)

    with patch("asyncio.sleep", fake_async_sleep), patch("time.sleep", fake_sleep):
        yield waits


METHODS = ["invoke", "ainvoke", "stream", "astream"]


@pytest.mark.parametrize("method", METHODS)
async def test_each_retry_path_waits_for_retry_after(sleeps, method):
    client = patch_with_retry(Client(RateLimited({"Retry-After": "3"})), retries=3, retry_codes=[429])
    assert await _call(client, method) in ("ok", ["ok"])
    assert sleeps == [3.0]


@pytest.mark.parametrize("method", METHODS)
async def test_without_retry_after_the_backoff_is_unchanged(sleeps, method):
    client = patch_with_retry(Client(RateLimited(), failures=2), retries=3, retry_codes=[429])
    assert await _call(client, method) in ("ok", ["ok"])
    assert sleeps == [0.25, 0.5]


async def test_retry_after_is_capped_by_max_wait(sleeps):
    client = patch_with_retry(Client(RateLimited({"Retry-After": "600"})), retries=3, retry_codes=[429], max_wait=60)
    assert await client.ainvoke() == "ok"
    assert sleeps == [60]


async def test_bedrock_throttling_waits_for_retry_after(sleeps):
    client = patch_with_retry(Client(Throttled({"retry-after": "2"})), retries=3, retry_on_messages=["Throttling"])
    assert await client.ainvoke() == "ok"
    assert sleeps == [2.0]


def test_retry_mixin_has_a_max_wait_with_a_60_second_default():
    from nat.data_models.retry_mixin import RetryMixin

    assert RetryMixin().retry_max_wait == 60.0
    with pytest.raises(ValueError):
        RetryMixin(retry_max_wait=-1)


@pytest.mark.parametrize("headers, expected", [({
    "Retry-After": "0"
}, 0.0), ({
    "retry-after-ms": "0"
}, 0.0)],
                         ids=["zero_seconds", "zero_ms"])
def test_a_zero_retry_after_means_no_wait(headers, expected):
    assert _retry_after_seconds(RateLimited(headers)) == expected


def test_odd_response_objects_have_no_retry_after():
    from unittest.mock import MagicMock

    error = RateLimited()
    error.response = MagicMock()
    assert _retry_after_seconds(error) is None


async def test_a_date_without_a_timezone_is_still_honored(sleeps):
    # RFC 2822 "-0000" parses to a naive datetime
    client = patch_with_retry(Client(RateLimited({"Retry-After": "Wed, 21 Oct 2099 07:28:00 -0000"})),
                              retries=3,
                              retry_codes=[429],
                              max_wait=7)
    assert await client.ainvoke() == "ok"
    assert sleeps == [7]


async def test_a_malformed_response_falls_back_to_backoff(sleeps):
    """Reading the header must never cost the retry itself."""
    error = RateLimited()
    error.response = {"ResponseMetadata": "not-a-dict"}
    client = patch_with_retry(Client(error), retries=3, retry_codes=[429])
    assert await client.ainvoke() == "ok"
    assert sleeps == [0.25]


async def test_max_wait_below_the_default_caps_the_wait(sleeps):
    client = patch_with_retry(Client(RateLimited({"Retry-After": "30"})), retries=3, retry_codes=[429], max_wait=5)
    assert await client.ainvoke() == "ok"
    assert sleeps == [5]


def test_retry_max_wait_rejects_infinity():
    from nat.data_models.retry_mixin import RetryMixin

    with pytest.raises(ValueError):
        RetryMixin(retry_max_wait=float("inf"))
