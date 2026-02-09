# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nat.builder.context import Context
from nat.profiler.decorators.latency import LatencySensitivity
from nat.profiler.decorators.latency import latency_sensitive


class TestLatencySensitivity:
    """Tests for LatencySensitivity enum."""

    def test_enum_values_exist(self):
        """Test that all three sensitivity levels exist."""
        assert LatencySensitivity.LOW
        assert LatencySensitivity.MEDIUM
        assert LatencySensitivity.HIGH

    def test_enum_string_values(self):
        """Test that enum values are correct strings."""
        assert LatencySensitivity.LOW.value == "LOW"
        assert LatencySensitivity.MEDIUM.value == "MEDIUM"
        assert LatencySensitivity.HIGH.value == "HIGH"

    def test_priority_values(self):
        """Test that priority property returns correct numeric values."""
        assert LatencySensitivity.LOW.priority == 1
        assert LatencySensitivity.MEDIUM.priority == 2
        assert LatencySensitivity.HIGH.priority == 3

    def test_parse_enum_value(self):
        """Test that parse accepts enum values directly."""
        result = LatencySensitivity.parse(LatencySensitivity.HIGH)
        assert result == LatencySensitivity.HIGH

    def test_parse_uppercase_string(self):
        """Test that parse accepts uppercase strings."""
        assert LatencySensitivity.parse("LOW") == LatencySensitivity.LOW
        assert LatencySensitivity.parse("MEDIUM") == LatencySensitivity.MEDIUM
        assert LatencySensitivity.parse("HIGH") == LatencySensitivity.HIGH

    def test_parse_lowercase_string(self):
        """Test that parse accepts lowercase strings (case-insensitive)."""
        assert LatencySensitivity.parse("low") == LatencySensitivity.LOW
        assert LatencySensitivity.parse("medium") == LatencySensitivity.MEDIUM
        assert LatencySensitivity.parse("high") == LatencySensitivity.HIGH

    def test_parse_mixed_case_string(self):
        """Test that parse accepts mixed-case strings."""
        assert LatencySensitivity.parse("Low") == LatencySensitivity.LOW
        assert LatencySensitivity.parse("MeDiUm") == LatencySensitivity.MEDIUM
        assert LatencySensitivity.parse("HiGh") == LatencySensitivity.HIGH

    def test_parse_invalid_string(self):
        """Test that parse raises ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):
            LatencySensitivity.parse("INVALID")

    def test_parse_invalid_type(self):
        """Test that parse raises ValueError for invalid types."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):
            LatencySensitivity.parse(123)

    def test_parse_empty_string(self):
        """Test that parse raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):
            LatencySensitivity.parse("")


class TestContextIntegration:
    """Tests for Context integration with latency sensitivity."""

    def test_default_sensitivity_is_medium(self):
        """Test that default latency sensitivity is MEDIUM."""
        ctx = Context.get()
        sensitivity = ctx.latency_sensitivity
        assert sensitivity == LatencySensitivity.MEDIUM

    def test_push_higher_priority_changes_sensitivity(self):
        """Test that pushing higher priority changes current sensitivity."""
        ctx = Context.get()

        # Default is MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        # Push HIGH (higher priority)
        with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
            assert ctx.latency_sensitivity == LatencySensitivity.HIGH

        # Reverts to MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_push_lower_priority_keeps_current(self):
        """Test that pushing lower priority keeps current sensitivity."""
        ctx = Context.get()

        # Push HIGH first
        with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
            assert ctx.latency_sensitivity == LatencySensitivity.HIGH

            # Try to push LOW (lower priority) - should stay HIGH
            with ctx.push_latency_sensitivity(LatencySensitivity.LOW):
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH

            # Still HIGH after inner context exits
            assert ctx.latency_sensitivity == LatencySensitivity.HIGH

        # Reverts to MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_deep_nesting_maintains_priority(self):
        """Test that deep nesting correctly maintains highest priority."""
        ctx = Context.get()

        # MEDIUM (default)
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        with ctx.push_latency_sensitivity(LatencySensitivity.LOW):
            # LOW < MEDIUM, stays MEDIUM
            assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

            with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
                # HIGH > MEDIUM, becomes HIGH
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                with ctx.push_latency_sensitivity(LatencySensitivity.MEDIUM):
                    # MEDIUM < HIGH, stays HIGH
                    assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                    with ctx.push_latency_sensitivity(LatencySensitivity.LOW):
                        # LOW < HIGH, stays HIGH
                        assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                    # Still HIGH
                    assert ctx.latency_sensitivity == LatencySensitivity.HIGH

                # Still HIGH
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH

            # Back to MEDIUM
            assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        # Back to MEDIUM
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_exception_in_context_still_pops(self):
        """Test that exceptions don't break stack management."""
        ctx = Context.get()

        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        try:
            with ctx.push_latency_sensitivity(LatencySensitivity.HIGH):
                assert ctx.latency_sensitivity == LatencySensitivity.HIGH
                raise ValueError("test error")
        except ValueError:
            pass

        # Should revert to MEDIUM despite exception
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM


class TestDecoratorSyncFunctions:
    """Tests for @latency_sensitive decorator on sync functions."""

    def test_sync_function_with_enum(self):
        """Test decorator on sync function with enum value."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def sync_func():
            return Context.get().latency_sensitivity

        # Outside decorator, should be MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

        # Inside decorator, should be HIGH
        result = sync_func()
        assert result == LatencySensitivity.HIGH

        # After decorator, back to MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

    def test_sync_function_with_string(self):
        """Test decorator on sync function with string value."""

        @latency_sensitive("low")
        def sync_func():
            return Context.get().latency_sensitivity

        result = sync_func()
        # LOW is in stack, but MEDIUM default has higher priority
        assert result == LatencySensitivity.MEDIUM

    def test_sync_function_priority_nesting(self):
        """Test priority-based nesting with sync functions."""

        @latency_sensitive(LatencySensitivity.LOW)
        def low_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(LatencySensitivity.HIGH)
        def high_func():
            inner = low_func()
            return Context.get().latency_sensitivity, inner

        outer, inner = high_func()
        # Both should be HIGH due to priority
        assert outer == LatencySensitivity.HIGH
        assert inner == LatencySensitivity.HIGH

    def test_sync_function_with_return_value(self):
        """Test that decorator preserves return values."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def func_with_return(x, y):
            return x + y

        result = func_with_return(2, 3)
        assert result == 5

    def test_sync_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def func_with_args(*args, **kwargs):
            return (args, kwargs)

        result = func_with_args(1, 2, 3, x=4, y=5)
        assert result == ((1, 2, 3), {"x": 4, "y": 5})

    def test_sync_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def failing_func():
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        with pytest.raises(ValueError, match="test error"):
            failing_func()

        # Should revert to MEDIUM despite exception
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_invalid_sensitivity_at_decoration_time(self):
        """Test that invalid sensitivity raises ValueError at decoration time."""
        with pytest.raises(ValueError, match="Invalid latency sensitivity"):

            @latency_sensitive("INVALID")
            def func():
                pass


class TestDecoratorAsyncFunctions:
    """Tests for @latency_sensitive decorator on async functions."""

    async def test_async_function_with_enum(self):
        """Test decorator on async function with enum value."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def async_func():
            return Context.get().latency_sensitivity

        # Outside decorator, should be MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

        # Inside decorator, should be HIGH
        result = await async_func()
        assert result == LatencySensitivity.HIGH

        # After decorator, back to MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

    async def test_async_function_with_string(self):
        """Test decorator on async function with string value."""

        @latency_sensitive("low")
        async def async_func():
            return Context.get().latency_sensitivity

        result = await async_func()
        # LOW is in stack, but MEDIUM default has higher priority
        assert result == LatencySensitivity.MEDIUM

    async def test_async_function_priority_nesting(self):
        """Test priority-based nesting with async functions."""

        @latency_sensitive(LatencySensitivity.LOW)
        async def low_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(LatencySensitivity.HIGH)
        async def high_func():
            inner = await low_func()
            return Context.get().latency_sensitivity, inner

        outer, inner = await high_func()
        # Both should be HIGH due to priority
        assert outer == LatencySensitivity.HIGH
        assert inner == LatencySensitivity.HIGH

    async def test_async_function_with_return_value(self):
        """Test that decorator preserves return values."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def func_with_return(x, y):
            await asyncio.sleep(0)  # Make it actually async
            return x + y

        result = await func_with_return(2, 3)
        assert result == 5

    async def test_async_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def func_with_args(*args, **kwargs):
            await asyncio.sleep(0)
            return (args, kwargs)

        result = await func_with_args(1, 2, 3, x=4, y=5)
        assert result == ((1, 2, 3), {"x": 4, "y": 5})

    async def test_async_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def failing_func():
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        with pytest.raises(ValueError, match="test error"):
            await failing_func()

        # Should revert to MEDIUM despite exception
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    async def test_mixed_sync_async_nesting(self):
        """Test that sync and async functions can nest together."""

        @latency_sensitive(LatencySensitivity.LOW)
        def sync_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(LatencySensitivity.HIGH)
        async def async_func():
            # HIGH takes precedence
            sync_result = sync_func()
            async_result = Context.get().latency_sensitivity
            return sync_result, async_result

        sync_result, async_result = await async_func()
        assert sync_result == LatencySensitivity.HIGH
        assert async_result == LatencySensitivity.HIGH


class TestDecoratorGeneratorFunctions:
    """Tests for @latency_sensitive decorator on generator functions."""

    def test_generator_function_with_enum(self):
        """Test decorator on generator function with enum value."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def gen_func():
            for i in range(3):
                yield (i, Context.get().latency_sensitivity)

        # Outside decorator, should be MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

        # Inside decorator, should be HIGH
        results = list(gen_func())
        assert len(results) == 3
        for i, sensitivity in results:
            assert sensitivity == LatencySensitivity.HIGH

        # After decorator, back to MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

    def test_generator_function_with_string(self):
        """Test decorator on generator function with string value."""

        @latency_sensitive("low")
        def gen_func():
            for i in range(2):
                yield Context.get().latency_sensitivity

        results = list(gen_func())
        # LOW is in stack, but MEDIUM default has higher priority
        assert all(s == LatencySensitivity.MEDIUM for s in results)

    def test_generator_function_priority_nesting(self):
        """Test priority-based nesting with generator functions."""

        @latency_sensitive(LatencySensitivity.LOW)
        def low_gen():
            yield Context.get().latency_sensitivity

        @latency_sensitive(LatencySensitivity.HIGH)
        def high_gen():
            # Get first value from low_gen while in HIGH context
            low_result = next(low_gen())
            yield Context.get().latency_sensitivity, low_result

        outer, inner = next(high_gen())
        # Both should be HIGH due to priority
        assert outer == LatencySensitivity.HIGH
        assert inner == LatencySensitivity.HIGH

    def test_generator_function_yields_values(self):
        """Test that decorator preserves yielded values."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def gen_with_values(n):
            for i in range(n):
                yield i * 2

        results = list(gen_with_values(4))
        assert results == [0, 2, 4, 6]

    def test_generator_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def gen_with_args(*args, **kwargs):
            yield args
            yield kwargs

        gen = gen_with_args(1, 2, 3, x=4, y=5)
        assert next(gen) == (1, 2, 3)
        assert next(gen) == {"x": 4, "y": 5}

    def test_generator_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def failing_gen():
            yield 1
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        gen = failing_gen()
        assert next(gen) == 1

        with pytest.raises(ValueError, match="test error"):
            next(gen)

        # Should revert to MEDIUM despite exception
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    def test_generator_function_early_exit(self):
        """Test that early exit from generator still pops stack."""

        @latency_sensitive(LatencySensitivity.HIGH)
        def gen_func():
            yield from range(10)

        ctx = Context.get()
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        # Only consume first 3 values
        gen = gen_func()
        results = [next(gen) for _ in range(3)]
        assert results == [0, 1, 2]

        # Close generator early
        gen.close()

        # Should still be able to access context after early exit
        # Note: Stack will pop when generator is garbage collected
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM


class TestDecoratorAsyncGeneratorFunctions:
    """Tests for @latency_sensitive decorator on async generator functions."""

    async def test_async_generator_function_with_enum(self):
        """Test decorator on async generator function with enum value."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def async_gen_func():
            for i in range(3):
                yield (i, Context.get().latency_sensitivity)

        # Outside decorator, should be MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

        # Inside decorator, should be HIGH
        results = [item async for item in async_gen_func()]
        assert len(results) == 3
        for i, sensitivity in results:
            assert sensitivity == LatencySensitivity.HIGH

        # After decorator, back to MEDIUM
        assert Context.get().latency_sensitivity == LatencySensitivity.MEDIUM

    async def test_async_generator_function_with_string(self):
        """Test decorator on async generator function with string value."""

        @latency_sensitive("low")
        async def async_gen_func():
            for i in range(2):
                yield Context.get().latency_sensitivity

        results = [item async for item in async_gen_func()]
        # LOW is in stack, but MEDIUM default has higher priority
        assert all(s == LatencySensitivity.MEDIUM for s in results)

    async def test_async_generator_function_priority_nesting(self):
        """Test priority-based nesting with async generator functions."""

        @latency_sensitive(LatencySensitivity.LOW)
        async def low_async_gen():
            yield Context.get().latency_sensitivity

        @latency_sensitive(LatencySensitivity.HIGH)
        async def high_async_gen():
            # Get first value from low_async_gen while in HIGH context
            async for val in low_async_gen():
                low_result = val
                break
            yield Context.get().latency_sensitivity, low_result

        async for outer, inner in high_async_gen():
            # Both should be HIGH due to priority
            assert outer == LatencySensitivity.HIGH
            assert inner == LatencySensitivity.HIGH

    async def test_async_generator_function_yields_values(self):
        """Test that decorator preserves yielded values."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def async_gen_with_values(n):
            for i in range(n):
                yield i * 2

        results = [item async for item in async_gen_with_values(4)]
        assert results == [0, 2, 4, 6]

    async def test_async_generator_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def async_gen_with_args(*args, **kwargs):
            yield args
            yield kwargs

        results = [item async for item in async_gen_with_args(1, 2, 3, x=4, y=5)]
        assert results[0] == (1, 2, 3)
        assert results[1] == {"x": 4, "y": 5}

    async def test_async_generator_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def failing_async_gen():
            yield 1
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        agen = failing_async_gen()
        assert await agen.__anext__() == 1

        with pytest.raises(ValueError, match="test error"):
            await agen.__anext__()

        # Should revert to MEDIUM despite exception
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    async def test_async_generator_function_early_exit(self):
        """Test that early exit from async generator still pops stack."""

        @latency_sensitive(LatencySensitivity.HIGH)
        async def async_gen_func():
            for i in range(10):
                yield i

        ctx = Context.get()
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

        # Only consume first 3 values
        agen = async_gen_func()
        results = []
        for _ in range(3):
            results.append(await agen.__anext__())
        assert results == [0, 1, 2]

        # Close async generator early
        await agen.aclose()

        # Should revert to MEDIUM after close
        assert ctx.latency_sensitivity == LatencySensitivity.MEDIUM

    async def test_mixed_async_and_async_gen_nesting(self):
        """Test that async functions and async generators can nest together."""

        @latency_sensitive(LatencySensitivity.LOW)
        async def async_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(LatencySensitivity.HIGH)
        async def high_async_gen():
            # HIGH takes precedence
            async_result = await async_func()
            gen_result = Context.get().latency_sensitivity
            yield async_result, gen_result

        async for async_result, gen_result in high_async_gen():
            assert async_result == LatencySensitivity.HIGH
            assert gen_result == LatencySensitivity.HIGH
