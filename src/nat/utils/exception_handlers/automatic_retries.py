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
import copy
import functools
import gc
import inspect
import logging
import re
import time
import types
import weakref
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import TypeVar

T = TypeVar("T")
Exc = tuple[type[BaseException], ...]  # exception classes
CodePattern = int | str | range  # for retry_codes argument
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Memory-optimized helpers
# ─────────────────────────────────────────────────────────────


def _shallow_copy_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Create shallow copies of args and kwargs to avoid deep copy overhead."""
    # For most use cases, shallow copy is sufficient and much faster
    return tuple(args), dict(kwargs)


def _deep_copy_args(args: tuple, kwargs: dict, skip_first: bool = False) -> tuple[tuple, dict]:
    """Create deep copies of args and kwargs to prevent mutation issues.

    Args:
        args: Positional arguments to copy
        kwargs: Keyword arguments to copy
        skip_first: If True, skip copying the first arg (typically 'self')
    """
    if skip_first and args:
        # Don't deep copy self, only the remaining arguments
        return (args[0], ) + copy.deepcopy(args[1:]), copy.deepcopy(kwargs)
    return copy.deepcopy(args), copy.deepcopy(kwargs)


def _clear_exception_context(exc: BaseException) -> None:
    """Clear exception traceback to free memory."""
    if exc is None:
        return

    # Clear the exception's traceback to break reference cycles
    # This is the main memory optimization
    try:
        exc.__traceback__ = None
    except AttributeError:
        pass

    # Also try to clear any chained exceptions
    try:
        if hasattr(exc, '__cause__') and exc.__cause__ is not None:
            _clear_exception_context(exc.__cause__)
        if hasattr(exc, '__context__') and exc.__context__ is not None:
            _clear_exception_context(exc.__context__)
    except AttributeError:
        pass


def _run_gc_if_needed(attempt: int, gc_frequency: int = 3) -> None:
    """Run garbage collection periodically to free memory."""
    if attempt > 0 and attempt % gc_frequency == 0:
        gc.collect()


# ─────────────────────────────────────────────────────────────
#  Helpers: status-code extraction & pattern matching
# ─────────────────────────────────────────────────────────────
_CODE_ATTRS = ("code", "status", "status_code", "http_status")


def _extract_status_code(exc: BaseException) -> int | None:
    """Return a numeric status code found inside *exc*, else None."""
    for attr in _CODE_ATTRS:
        if hasattr(exc, attr):
            try:
                return int(getattr(exc, attr))
            except (TypeError, ValueError):
                pass
    if exc.args:
        try:
            return int(exc.args[0])
        except (TypeError, ValueError):
            pass
    return None


def _pattern_to_regex(pat: str) -> re.Pattern[str]:
    """
    Convert simple wildcard pattern ("4xx", "5*", "40x") to a ^regex$.
    Rule:  'x' or '*' ⇒ any digit.
    """
    escaped = re.escape(pat)
    regex_pattern = escaped.replace(r"\*", r"\d").replace("x", r"\d")
    return re.compile("^" + regex_pattern + "$")


def _code_matches(code: int, pat: CodePattern) -> bool:
    if isinstance(pat, int):
        return code == pat
    if isinstance(pat, range):
        return code in pat
    return bool(_pattern_to_regex(pat).match(str(code)))


# ─────────────────────────────────────────────────────────────
#  Unified retry-decision helper (unchanged)
# ─────────────────────────────────────────────────────────────
def _want_retry(
    exc: BaseException,
    *,
    code_patterns: Sequence[CodePattern] | None,
    msg_substrings: Sequence[str] | None,
) -> bool:
    """
    Return True if the exception satisfies *either* (when provided):
       • code_patterns  – matches status-code pattern(s)
       • msg_substrings – contains any of the substrings (case-insensitive)
    """

    if not code_patterns and not msg_substrings:
        logger.info("Retrying on exception %s without extra filters", exc)
        return True

    # -------- status-code filter --------
    if code_patterns is not None:
        code = _extract_status_code(exc)
        if any(_code_matches(code, p) for p in code_patterns):
            logger.info("Retrying on exception %s with matched code %s", exc, code)
            return True

    # -------- message filter -----------
    if msg_substrings is not None:
        msg = str(exc).lower()
        if any(s.lower() in msg for s in msg_substrings):
            logger.info("Retrying on exception %s with matched message %s", exc, msg)
            return True

    return False


# ─────────────────────────────────────────────────────────────
#  Memory-optimized decorator factory
# ─────────────────────────────────────────────────────────────
def _retry_decorator(
    *,
    retries: int = 3,
    base_delay: float = 0.25,
    backoff: float = 2.0,
    retry_on: Exc = (Exception, ),
    retry_codes: Sequence[CodePattern] | None = None,
    retry_on_messages: Sequence[str] | None = None,
    shallow_copy: bool = True,  # Changed default to shallow copy
    gc_frequency: int = 3,  # Run GC every N retries
    clear_tracebacks: bool = True,  # Clear exception tracebacks
    instance_context_aware: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """"
    Build a decorator that retries with exponential back-off *iff*:

      • the raised exception is an instance of one of `retry_on`
      • AND `_want_retry()` returns True (i.e. matches codes/messages filters)

    If both `retry_codes` and `retry_on_messages` are None, all exceptions are retried.

    instance_context_aware:
        If True, the decorator will check for a retry context flag on the first
        argument (assumed to be 'self'). If the flag is set, retries are skipped
        to prevent retry storms in nested method calls.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        use_shallow_copy = shallow_copy
        use_context_aware = instance_context_aware
        skip_self_in_deepcopy = instance_context_aware

        class _RetryContext:
            """Context manager for instance-level retry gating."""

            __slots__ = ("_obj_ref", "_enabled", "_active")

            def __init__(self, args: tuple[Any, ...]):
                if use_context_aware and args:
                    try:
                        # Use weak reference to avoid keeping objects alive
                        self._obj_ref = weakref.ref(args[0])
                        self._enabled = True
                    except TypeError:
                        # Object doesn't support weak references
                        self._obj_ref = None
                        self._enabled = False
                else:
                    self._obj_ref = None
                    self._enabled = False
                self._active = False

            def __enter__(self):
                if not self._enabled or self._obj_ref is None:
                    return False

                obj = self._obj_ref()
                if obj is None:
                    return False

                try:
                    # If already in retry context, skip retries
                    if getattr(obj, "_in_retry_context", False):
                        return True
                    object.__setattr__(obj, "_in_retry_context", True)
                    self._active = True
                    return False
                except Exception:
                    # Cannot set attribute, disable context
                    self._enabled = False
                    return False

            def __exit__(self, _exc_type, _exc, _tb):
                if (self._enabled and self._active and self._obj_ref is not None):
                    obj = self._obj_ref()
                    if obj is not None:
                        try:
                            object.__setattr__(obj, "_in_retry_context", False)
                        except Exception:
                            pass

        async def _call_with_retry_async(*args, **kw) -> T:
            with _RetryContext(args) as already_in_context:
                if already_in_context:
                    return await fn(*args, **kw)

                delay = base_delay
                last_exception = None

                for attempt in range(retries):
                    # Copy args based on configuration
                    if use_shallow_copy:
                        call_args, call_kwargs = _shallow_copy_args(args, kw)
                    else:
                        call_args, call_kwargs = _deep_copy_args(args, kw, skip_first=skip_self_in_deepcopy)

                    try:
                        return await fn(*call_args, **call_kwargs)
                    except retry_on as exc:
                        last_exception = exc

                        # Clear traceback to free memory
                        if clear_tracebacks:
                            _clear_exception_context(exc)

                        # Run GC periodically
                        _run_gc_if_needed(attempt, gc_frequency)

                        if not _want_retry(exc, code_patterns=retry_codes,
                                           msg_substrings=retry_on_messages) or attempt == retries - 1:
                            raise

                        await asyncio.sleep(delay)
                        delay *= backoff

                if last_exception:
                    raise last_exception

        async def _agen_with_retry(*args, **kw):
            with _RetryContext(args) as already_in_context:
                if already_in_context:
                    async for item in fn(*args, **kw):
                        yield item
                    return

                delay = base_delay
                last_exception = None

                for attempt in range(retries):
                    if use_shallow_copy:
                        call_args, call_kwargs = _shallow_copy_args(args, kw)
                    else:
                        call_args, call_kwargs = _deep_copy_args(args, kw, skip_first=skip_self_in_deepcopy)

                    try:
                        async for item in fn(*call_args, **call_kwargs):
                            yield item
                        return
                    except retry_on as exc:
                        last_exception = exc

                        # Memory cleanup
                        if clear_tracebacks:
                            _clear_exception_context(exc)

                        _run_gc_if_needed(attempt, gc_frequency)

                        if not _want_retry(exc, code_patterns=retry_codes,
                                           msg_substrings=retry_on_messages) or attempt == retries - 1:
                            raise

                        await asyncio.sleep(delay)
                        delay *= backoff

                if last_exception:
                    raise last_exception

        def _gen_with_retry(*args, **kw) -> Iterable[Any]:
            with _RetryContext(args) as already_in_context:
                if already_in_context:
                    yield from fn(*args, **kw)
                    return

                delay = base_delay
                last_exception = None

                for attempt in range(retries):
                    if use_shallow_copy:
                        call_args, call_kwargs = _shallow_copy_args(args, kw)
                    else:
                        call_args, call_kwargs = _deep_copy_args(args, kw, skip_first=skip_self_in_deepcopy)

                    try:
                        yield from fn(*call_args, **call_kwargs)
                        return
                    except retry_on as exc:
                        last_exception = exc

                        # Memory cleanup
                        if clear_tracebacks:
                            _clear_exception_context(exc)

                        _run_gc_if_needed(attempt, gc_frequency)

                        if not _want_retry(exc, code_patterns=retry_codes,
                                           msg_substrings=retry_on_messages) or attempt == retries - 1:
                            raise

                        time.sleep(delay)
                        delay *= backoff

                if last_exception:
                    raise last_exception

        def _sync_with_retry(*args, **kw) -> T:
            with _RetryContext(args) as already_in_context:
                if already_in_context:
                    return fn(*args, **kw)

                delay = base_delay
                last_exception = None

                for attempt in range(retries):
                    if use_shallow_copy:
                        call_args, call_kwargs = _shallow_copy_args(args, kw)
                    else:
                        call_args, call_kwargs = _deep_copy_args(args, kw, skip_first=skip_self_in_deepcopy)

                    try:
                        return fn(*call_args, **call_kwargs)
                    except retry_on as exc:
                        last_exception = exc

                        # Memory cleanup
                        if clear_tracebacks:
                            _clear_exception_context(exc)

                        _run_gc_if_needed(attempt, gc_frequency)

                        if not _want_retry(exc, code_patterns=retry_codes,
                                           msg_substrings=retry_on_messages) or attempt == retries - 1:
                            raise

                        time.sleep(delay)
                        delay *= backoff

                if last_exception:
                    raise last_exception

        # Decide which wrapper to return
        if inspect.iscoroutinefunction(fn):
            wrapper = _call_with_retry_async
        elif inspect.isasyncgenfunction(fn):
            wrapper = _agen_with_retry
        elif inspect.isgeneratorfunction(fn):
            wrapper = _gen_with_retry
        else:
            wrapper = _sync_with_retry

        return functools.wraps(fn)(wrapper)  # type: ignore[return-value]

    return decorate


def patch_with_retry(
    obj: Any,
    *,
    retries: int = 3,
    base_delay: float = 0.25,
    backoff: float = 2.0,
    retry_on: Exc = (Exception, ),
    retry_codes: Sequence[CodePattern] | None = None,
    retry_on_messages: Sequence[str] | None = None,
    deep_copy: bool = False,
    gc_frequency: int = 3,
    clear_tracebacks: bool = True,
) -> Any:
    """
    Patch *obj* instance-locally so **every public method** retries on failure.

    Extra filters
    -------------
    retry_codes
        Same as before – ints, ranges, or wildcard strings (“4xx”, “5*”…).
    retry_on_messages
        List of *substring* patterns.  We retry only if **any** pattern
        appears (case-insensitive) in `str(exc)`.
    deepcopy:
        If True, each retry receives deep‑copied *args and **kwargs* to avoid
        mutating shared state between attempts.
    """

    # Invert deep copy to keep function signature the same
    shallow_copy = not deep_copy

    deco = _retry_decorator(
        retries=retries,
        base_delay=base_delay,
        backoff=backoff,
        retry_on=retry_on,
        retry_codes=retry_codes,
        retry_on_messages=retry_on_messages,
        shallow_copy=shallow_copy,
        gc_frequency=gc_frequency,
        clear_tracebacks=clear_tracebacks,
        instance_context_aware=True,  # Prevent retry storms
    )

    # Choose attribute source: the *class* to avoid __getattr__
    cls = obj if inspect.isclass(obj) else type(obj)
    cls_name = getattr(cls, "__name__", str(cls))

    for name, _ in inspect.getmembers(cls, callable):
        descriptor = inspect.getattr_static(cls, name)

        # Skip dunders, privates and all descriptors we must not wrap
        if name.startswith("_") or isinstance(descriptor, property | staticmethod | classmethod):
            continue

        original = descriptor.__func__ if isinstance(descriptor, types.MethodType) else descriptor
        wrapped = deco(original)

        try:  # instance‑level first
            if not inspect.isclass(obj):
                object.__setattr__(obj, name, types.MethodType(wrapped, obj))
                continue
        except Exception as exc:
            logger.info(
                "Instance‑level patch failed for %s.%s (%s); "
                "falling back to class‑level patch.",
                cls_name,
                name,
                exc,
            )

        try:  # class‑level fallback
            setattr(cls, name, wrapped)
        except Exception as exc:
            logger.info(
                "Cannot patch method %s.%s with automatic retries: %s",
                cls_name,
                name,
                exc,
            )

    return obj
