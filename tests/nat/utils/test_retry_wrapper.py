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
import asyncio
from collections.abc import Iterable

import pytest

from aiq.utils.exception_handlers import automatic_retries as ar

# Helpers --------------------------------------------------------------------


class APIError(Exception):
    """
    Lightweight HTTP‑style error for tests.

    Parameters
    ----------
    code:
        Numeric status code (e.g. 503).
    msg:
        Optional human‑readable description.  If omitted, a default
        message ``"HTTP {code}"`` is used.
    """

    def __init__(self, code: int, msg: str = ""):
        self.code = code
        super().__init__(msg or f"HTTP {code}")


# ---------------------------------------------------------------------------
# 1. _unit_ tests for _want_retry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "code_patterns,msg_patterns,exc,expected",
    [
        # --- no filters supplied -> always retry ---------------------------
        (None, None, Exception("irrelevant"), True),
        # --- code filter only ---------------------------------------------
        (["4xx"], None, APIError(404), True),
        (["4xx"], None, APIError(500), False),
        ([429, range(500, 510)], None, APIError(429), True),
        ([429, range(500, 510)], None, APIError(503), True),
        # --- message filter only ------------------------------------------
        (None, ["timeout", "temporarily unavailable"], APIError(200, "Timeout"), True),
        (None, ["timeout"], APIError(200, "Something else"), False),
        # --- both filters present (OR logic) ------------------------------
        (["5xx"], ["unavailable"], APIError(503, "no match"), True),  # code matches
        (["4xx"], ["unavailable"], APIError(503, "Service unavailable"), True),  # msg matches
        (["4xx"], ["bad"], APIError(503, "Service unavailable"), False),  # none match
    ],
)
def test_want_retry(code_patterns, msg_patterns, exc, expected):
    """Exhaustively validate `_want_retry` for every branch:

    * No filters provided  -> always True
    * Code‑only filtering  -> match / no‑match
    * Message‑only filter  -> match / no‑match
    * Combined filters     -> OR logic
    """
    assert (ar._want_retry(
        exc,
        code_patterns=code_patterns,
        msg_substrings=msg_patterns,
    ) is expected)


# ---------------------------------------------------------------------------
# 2. integration tests for patch_with_retry (sync / async / gen)
# ---------------------------------------------------------------------------
class Service:
    """
    Toy service whose methods fail exactly once and then succeed.

    The counters (`calls_sync`, `calls_gen`, `calls_async`) make it easy
    to assert how many attempts were made, thereby confirming whether
    retry logic was invoked.
    """

    def __init__(self):
        self.calls_sync = 0
        self.calls_gen = 0
        self.calls_async = 0

    # ---- plain sync -------------------------------------------------------
    def sync_method(self):
        """Synchronous function that raises once, then returns 'sync‑ok'."""
        self.calls_sync += 1
        if self.calls_sync < 2:  # fail the first call
            raise APIError(503, "Service unavailable")
        return "sync-ok"

    # ---- sync generator ---------------------------------------------------
    def gen_method(self) -> Iterable[int]:
        """Sync generator that raises once, then yields 0,1,2."""
        self.calls_gen += 1
        if self.calls_gen < 2:
            raise APIError(429, "Too Many Requests")
        yield from range(3)

    # ---- async coroutine --------------------------------------------------
    async def async_method(self):
        """Async coroutine that raises once, then returns 'async‑ok'."""
        self.calls_async += 1
        if self.calls_async < 2:
            raise APIError(500, "Server exploded")
        return "async-ok"


# monkey-patch time.sleep / asyncio.sleep so tests run instantly -------------
@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Fixture that monkey‑patches blocking sleeps with no‑ops.

    Eliminates real delays so the test suite executes near‑instantaneously.
    """
    # Patch time.sleep with a synchronous no‑op.
    monkeypatch.setattr(ar.time, "sleep", lambda *_: None)

    # Create an async no‑op to replace asyncio.sleep.
    async def _async_noop(*_args, **_kw):
        return None

    # Patch both the automatic_retries asyncio reference and the global asyncio.
    monkeypatch.setattr(ar.asyncio, "sleep", _async_noop)
    monkeypatch.setattr(asyncio, "sleep", _async_noop)


def _patch_service(**kwargs):
    """Return a freshly wrapped `Service` instance with default retry settings."""
    svc = Service()
    return ar.patch_with_retry(
        svc,
        retries=3,
        base_delay=0,  # avoid real sleep even if monkeypatch fails
        retry_codes=["4xx", "5xx", 429],
        **kwargs,
    )


def test_patch_preserves_type():
    """Ensure `patch_with_retry` does not alter the instance's type or identity."""
    svc = _patch_service()
    assert isinstance(svc, Service)
    assert svc.sync_method.__self__ is svc


def test_sync_retry():
    """Verify that a plain sync method retries exactly once and then succeeds."""
    svc = _patch_service()
    assert svc.sync_method() == "sync-ok"
    # first call raised, second succeeded
    assert svc.calls_sync == 2


def test_generator_retry():
    """Verify that a sync‑generator method retries, then yields all expected items."""
    svc = _patch_service()
    assert list(svc.gen_method()) == [0, 1, 2]
    assert svc.calls_gen == 2


async def test_async_retry():
    """Verify that an async coroutine retries exactly once and then succeeds."""
    svc = _patch_service()
    assert await svc.async_method() == "async-ok"
    assert svc.calls_async == 2
