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

from nat.utils.exception_handlers import automatic_retries as ar

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


# ---------------------------------------------------------------------------
# 3. Tests for nested retry prevention (retry storm prevention)
# ---------------------------------------------------------------------------
class NestedService:
    """Service with methods that call each other to test retry storm prevention."""

    def __init__(self):
        self.outer_calls = 0
        self.inner_calls = 0
        self.deep_calls = 0
        self.outer_failures = 1  # How many times outer should fail
        self.inner_failures = 1  # How many times inner should fail

    def outer_method(self):
        """Outer method that calls inner_method."""
        self.outer_calls += 1
        if self.outer_calls <= self.outer_failures:
            raise APIError(503, "Outer failed")
        # Call inner method - this should NOT retry if outer is already retrying
        return f"outer({self.inner_method()})"

    def inner_method(self):
        """Inner method that may fail."""
        self.inner_calls += 1
        if self.inner_calls <= self.inner_failures:
            raise APIError(503, "Inner failed")
        return "inner-ok"

    def deep_method(self):
        """Method that calls outer_method for deep nesting test."""
        self.deep_calls += 1
        if self.deep_calls <= 1:
            raise APIError(503, "Deep failed")
        return f"deep({self.outer_method()})"

    async def async_outer(self):
        """Async outer method that calls async inner."""
        self.outer_calls += 1
        if self.outer_calls <= self.outer_failures:
            raise APIError(503, "Async outer failed")
        result = await self.async_inner()
        return f"async-outer({result})"

    async def async_inner(self):
        """Async inner method that may fail."""
        self.inner_calls += 1
        if self.inner_calls <= self.inner_failures:
            raise APIError(503, "Async inner failed")
        return "async-inner-ok"


def test_nested_retry_prevention():
    """Test that nested method calls don't cause retry storms."""
    svc = NestedService()
    svc = ar.patch_with_retry(
        svc,
        retries=3,
        base_delay=0,
        retry_codes=["5xx"],
    )

    # Both methods fail once, then succeed
    svc.outer_failures = 1
    svc.inner_failures = 1

    result = svc.outer_method()
    assert result == "outer(inner-ok)"

    # Without retry storm prevention, we'd see:
    # - outer tries and fails, inner tries and fails
    # - outer retries (attempt 2), inner is called again and retries too
    # This would result in inner_calls = 4 (2 attempts × 2 retries)

    # With retry storm prevention:
    # - outer tries and fails (outer_calls = 1)
    # - outer retries (outer_calls = 2), calls inner
    # - inner is already in retry context, so it doesn't retry (inner_calls = 2)

    assert svc.outer_calls == 3  # Initial + 1 retry
    assert svc.inner_calls == 2  # Called twice, but no nested retries


def test_inner_method_can_still_retry_when_called_directly():
    """Test that inner methods can still retry when called directly (not nested)."""
    svc = NestedService()
    svc = ar.patch_with_retry(
        svc,
        retries=3,
        base_delay=0,
        retry_codes=["5xx"],
    )

    svc.inner_failures = 2  # Fail twice before succeeding

    # Call inner directly - it should retry normally
    result = svc.inner_method()
    assert result == "inner-ok"
    assert svc.inner_calls == 3  # Initial + 2 retries


def test_deep_nesting():
    """Test retry prevention with 3 levels of nesting."""
    svc = NestedService()
    svc = ar.patch_with_retry(
        svc,
        retries=3,
        base_delay=0,
        retry_codes=["5xx"],
    )

    # Make deep fail once, but inner and outer succeed on first try
    svc.outer_failures = 0  # outer always succeeds
    svc.inner_failures = 0  # inner always succeeds

    result = svc.deep_method()
    assert result == "deep(outer(inner-ok))"

    # Only deep_method should retry, others should execute without retries
    assert svc.deep_calls == 2  # Initial + 1 retry
    assert svc.outer_calls == 1
    assert svc.inner_calls == 1


def test_retry_storm_prevention_with_all_methods_failing():
    """Test that demonstrates retry storm prevention when inner fails multiple times."""
    svc = NestedService()
    svc = ar.patch_with_retry(
        svc,
        retries=3,
        base_delay=0,
        retry_codes=["5xx"],
    )

    # Set up so that inner needs multiple attempts to succeed
    svc.outer_failures = 0  # outer succeeds every time
    svc.inner_failures = 2  # inner fails first 2 times

    result = svc.outer_method()
    assert result == "outer(inner-ok)"

    # Without retry storm prevention, inner_calls could be much higher
    # With prevention: outer attempts three times total and calls inner each time
    # inner fails first 2 times, succeeds on 3rd
    assert svc.outer_calls == 3  # Called once per attempt
    assert svc.inner_calls == 3  # Called once per outer attempt

    # The key point: inner_calls is NOT 9 (3 outer attempts × 3 inner retries each)
    # which would happen without retry storm prevention


async def test_async_nested_retry_prevention():
    """Test that nested async method calls don't cause retry storms."""
    svc = NestedService()
    svc = ar.patch_with_retry(
        svc,
        retries=3,
        base_delay=0,
        retry_codes=["5xx"],
    )

    # Reset counters for async test
    svc.outer_calls = 0
    svc.inner_calls = 0
    svc.outer_failures = 1
    svc.inner_failures = 1

    result = await svc.async_outer()
    assert result == "async-outer(async-inner-ok)"

    # Same as sync test - no retry storms
    assert svc.outer_calls == 3  # Initial + 1 retry
    assert svc.inner_calls == 2  # Called twice, but no nested retries


def test_multiple_instances_dont_interfere():
    """Test that retry context is instance-specific."""
    svc1 = NestedService()
    svc2 = NestedService()

    svc1 = ar.patch_with_retry(svc1, retries=3, base_delay=0, retry_codes=["5xx"])
    svc2 = ar.patch_with_retry(svc2, retries=3, base_delay=0, retry_codes=["5xx"])

    # Both instances should retry independently
    svc1.inner_failures = 2
    svc2.inner_failures = 2

    result1 = svc1.inner_method()
    result2 = svc2.inner_method()

    assert result1 == "inner-ok"
    assert result2 == "inner-ok"
    assert svc1.inner_calls == 3  # Each instance retries independently
    assert svc2.inner_calls == 3


def test_exception_propagation_in_nested_calls():
    """Test that exceptions still propagate correctly in nested calls."""
    svc = NestedService()
    svc = ar.patch_with_retry(
        svc,
        retries=2,  # Only 2 retries
        base_delay=0,
        retry_codes=["5xx"],
    )

    # Inner method will fail 3 times (more than retry count)
    svc.outer_failures = 0  # Outer succeeds
    svc.inner_failures = 3  # Inner fails 3 times

    with pytest.raises(APIError) as exc_info:
        svc.outer_method()

    assert exc_info.value.code == 503
    assert "Inner failed" in str(exc_info.value)

    # Outer should be called twice (initial + 1 retry)
    # Inner should be called twice (once per outer call, no nested retries)
    assert svc.outer_calls == 2
    assert svc.inner_calls == 2
