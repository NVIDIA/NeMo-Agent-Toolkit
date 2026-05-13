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
from nat.plugins.profiler.decorators.latency import latency_sensitive


class TestLatencySensitiveValidation:
    """Tests for latency_sensitive decorator input validation."""

    def test_accepts_int(self):
        """Test that @latency_sensitive accepts an integer."""

        @latency_sensitive(3)
        def sync_func():
            return Context.get().latency_sensitivity

        result = sync_func()
        assert result == 3

    def test_rejects_string(self):
        """Test that @latency_sensitive rejects a string."""
        with pytest.raises(TypeError):

            @latency_sensitive("high")
            def sync_func():
                pass

    def test_rejects_float(self):
        """Test that @latency_sensitive rejects a float."""
        with pytest.raises(TypeError):

            @latency_sensitive(3.0)
            def sync_func():
                pass

    def test_rejects_none(self):
        """Test that @latency_sensitive rejects None."""
        with pytest.raises(TypeError):

            @latency_sensitive(None)
            def sync_func():
                pass

    def test_accepts_zero(self):
        """Test that @latency_sensitive accepts zero."""

        @latency_sensitive(0)
        def sync_func():
            return Context.get().latency_sensitivity

        # 0 < default 2, so default wins
        result = sync_func()
        assert result == 2

    def test_accepts_negative(self):
        """Test that @latency_sensitive accepts a negative integer."""

        @latency_sensitive(-1)
        def sync_func():
            return Context.get().latency_sensitivity

        # -1 < default 2, so default wins
        result = sync_func()
        assert result == 2

    def test_accepts_large_int(self):
        """Test that @latency_sensitive accepts a large integer."""

        @latency_sensitive(100)
        def sync_func():
            return Context.get().latency_sensitivity

        result = sync_func()
        assert result == 100

    def test_accepts_arbitrary_int(self):
        """Test that @latency_sensitive accepts an arbitrary integer like 42."""

        @latency_sensitive(42)
        def sync_func():
            return Context.get().latency_sensitivity

        result = sync_func()
        # 42 > default 2, so 42 wins
        assert result == 42


class TestContextIntegration:
    """Tests for Context integration with latency sensitivity."""

    def test_default_sensitivity_is_medium(self):
        """Test that default latency sensitivity is 2 (MEDIUM)."""
        ctx = Context.get()
        sensitivity = ctx.latency_sensitivity
        assert sensitivity == 2

    def test_push_higher_priority_changes_sensitivity(self):
        """Test that pushing higher priority changes current sensitivity."""
        ctx = Context.get()

        # Default is 2
        assert ctx.latency_sensitivity == 2

        # Push 3 (HIGH, higher priority)
        with ctx.push_latency_sensitivity(3):
            assert ctx.latency_sensitivity == 3

        # Reverts to 2
        assert ctx.latency_sensitivity == 2

    def test_push_lower_priority_keeps_current(self):
        """Test that pushing lower priority keeps current sensitivity."""
        ctx = Context.get()

        # Push 3 (HIGH) first
        with ctx.push_latency_sensitivity(3):
            assert ctx.latency_sensitivity == 3

            # Try to push 1 (LOW, lower priority) - should stay 3
            with ctx.push_latency_sensitivity(1):
                assert ctx.latency_sensitivity == 3

            # Still 3 after inner context exits
            assert ctx.latency_sensitivity == 3

        # Reverts to 2
        assert ctx.latency_sensitivity == 2

    def test_deep_nesting_maintains_priority(self):
        """Test that deep nesting correctly maintains highest priority."""
        ctx = Context.get()

        # 2 (default)
        assert ctx.latency_sensitivity == 2

        with ctx.push_latency_sensitivity(1):
            # 1 < 2, stays 2
            assert ctx.latency_sensitivity == 2

            with ctx.push_latency_sensitivity(3):
                # 3 > 2, becomes 3
                assert ctx.latency_sensitivity == 3

                with ctx.push_latency_sensitivity(2):
                    # 2 < 3, stays 3
                    assert ctx.latency_sensitivity == 3

                    with ctx.push_latency_sensitivity(1):
                        # 1 < 3, stays 3
                        assert ctx.latency_sensitivity == 3

                    # Still 3
                    assert ctx.latency_sensitivity == 3

                # Still 3
                assert ctx.latency_sensitivity == 3

            # Back to 2
            assert ctx.latency_sensitivity == 2

        # Back to 2
        assert ctx.latency_sensitivity == 2

    def test_exception_in_context_still_pops(self):
        """Test that exceptions don't break stack management."""
        ctx = Context.get()

        assert ctx.latency_sensitivity == 2

        try:
            with ctx.push_latency_sensitivity(3):
                assert ctx.latency_sensitivity == 3
                raise ValueError("test error")
        except ValueError:
            pass

        # Should revert to 2 despite exception
        assert ctx.latency_sensitivity == 2


class TestDecoratorSyncFunctions:
    """Tests for @latency_sensitive decorator on sync functions."""

    def test_sync_function_with_int(self):
        """Test decorator on sync function with integer value."""

        @latency_sensitive(3)
        def sync_func():
            return Context.get().latency_sensitivity

        # Outside decorator, should be 2
        assert Context.get().latency_sensitivity == 2

        # Inside decorator, should be 3
        result = sync_func()
        assert result == 3

        # After decorator, back to 2
        assert Context.get().latency_sensitivity == 2

    def test_sync_function_with_lower_int(self):
        """Test decorator on sync function with lower integer value."""

        @latency_sensitive(1)
        def sync_func():
            return Context.get().latency_sensitivity

        result = sync_func()
        # 1 is in stack, but default 2 has higher priority
        assert result == 2

    def test_sync_function_priority_nesting(self):
        """Test priority-based nesting with sync functions."""

        @latency_sensitive(1)
        def low_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(3)
        def high_func():
            inner = low_func()
            return Context.get().latency_sensitivity, inner

        outer, inner = high_func()
        # Both should be 3 due to priority
        assert outer == 3
        assert inner == 3

    def test_sync_function_with_return_value(self):
        """Test that decorator preserves return values."""

        @latency_sensitive(3)
        def func_with_return(x, y):
            return x + y

        result = func_with_return(2, 3)
        assert result == 5

    def test_sync_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(3)
        def func_with_args(*args, **kwargs):
            return (args, kwargs)

        result = func_with_args(1, 2, 3, x=4, y=5)
        assert result == ((1, 2, 3), {"x": 4, "y": 5})

    def test_sync_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(3)
        def failing_func():
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == 2

        with pytest.raises(ValueError, match="test error"):
            failing_func()

        # Should revert to 2 despite exception
        assert ctx.latency_sensitivity == 2

    def test_invalid_sensitivity_at_decoration_time(self):
        """Test that invalid sensitivity raises TypeError at decoration time."""
        with pytest.raises(TypeError):

            @latency_sensitive("INVALID")
            def func():
                pass


class TestDecoratorAsyncFunctions:
    """Tests for @latency_sensitive decorator on async functions."""

    async def test_async_function_with_int(self):
        """Test decorator on async function with integer value."""

        @latency_sensitive(3)
        async def async_func():
            return Context.get().latency_sensitivity

        # Outside decorator, should be 2
        assert Context.get().latency_sensitivity == 2

        # Inside decorator, should be 3
        result = await async_func()
        assert result == 3

        # After decorator, back to 2
        assert Context.get().latency_sensitivity == 2

    async def test_async_function_with_lower_int(self):
        """Test decorator on async function with lower integer value."""

        @latency_sensitive(1)
        async def async_func():
            return Context.get().latency_sensitivity

        result = await async_func()
        # 1 is in stack, but default 2 has higher priority
        assert result == 2

    async def test_async_function_priority_nesting(self):
        """Test priority-based nesting with async functions."""

        @latency_sensitive(1)
        async def low_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(3)
        async def high_func():
            inner = await low_func()
            return Context.get().latency_sensitivity, inner

        outer, inner = await high_func()
        # Both should be 3 due to priority
        assert outer == 3
        assert inner == 3

    async def test_async_function_with_return_value(self):
        """Test that decorator preserves return values."""

        @latency_sensitive(3)
        async def func_with_return(x, y):
            await asyncio.sleep(0)  # Make it actually async
            return x + y

        result = await func_with_return(2, 3)
        assert result == 5

    async def test_async_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(3)
        async def func_with_args(*args, **kwargs):
            await asyncio.sleep(0)
            return (args, kwargs)

        result = await func_with_args(1, 2, 3, x=4, y=5)
        assert result == ((1, 2, 3), {"x": 4, "y": 5})

    async def test_async_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(3)
        async def failing_func():
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == 2

        with pytest.raises(ValueError, match="test error"):
            await failing_func()

        # Should revert to 2 despite exception
        assert ctx.latency_sensitivity == 2

    async def test_mixed_sync_async_nesting(self):
        """Test that sync and async functions can nest together."""

        @latency_sensitive(1)
        def sync_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(3)
        async def async_func():
            # 3 takes precedence
            sync_result = sync_func()
            async_result = Context.get().latency_sensitivity
            return sync_result, async_result

        sync_result, async_result = await async_func()
        assert sync_result == 3
        assert async_result == 3


class TestDecoratorGeneratorFunctions:
    """Tests for @latency_sensitive decorator on generator functions."""

    def test_generator_function_with_int(self):
        """Test decorator on generator function with integer value."""

        @latency_sensitive(3)
        def gen_func():
            for i in range(3):
                yield (i, Context.get().latency_sensitivity)

        # Outside decorator, should be 2
        assert Context.get().latency_sensitivity == 2

        # Inside decorator, should be 3
        results = list(gen_func())
        assert len(results) == 3
        for i, sensitivity in results:
            assert sensitivity == 3

        # After decorator, back to 2
        assert Context.get().latency_sensitivity == 2

    def test_generator_function_with_lower_int(self):
        """Test decorator on generator function with lower integer value."""

        @latency_sensitive(1)
        def gen_func():
            for i in range(2):
                yield Context.get().latency_sensitivity

        results = list(gen_func())
        # 1 is in stack, but default 2 has higher priority
        assert all(s == 2 for s in results)

    def test_generator_function_priority_nesting(self):
        """Test priority-based nesting with generator functions."""

        @latency_sensitive(1)
        def low_gen():
            yield Context.get().latency_sensitivity

        @latency_sensitive(3)
        def high_gen():
            # Get first value from low_gen while in 3 context
            low_result = next(low_gen())
            yield Context.get().latency_sensitivity, low_result

        outer, inner = next(high_gen())
        # Both should be 3 due to priority
        assert outer == 3
        assert inner == 3

    def test_generator_function_yields_values(self):
        """Test that decorator preserves yielded values."""

        @latency_sensitive(3)
        def gen_with_values(n):
            for i in range(n):
                yield i * 2

        results = list(gen_with_values(4))
        assert results == [0, 2, 4, 6]

    def test_generator_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(3)
        def gen_with_args(*args, **kwargs):
            yield args
            yield kwargs

        gen = gen_with_args(1, 2, 3, x=4, y=5)
        assert next(gen) == (1, 2, 3)
        assert next(gen) == {"x": 4, "y": 5}

    def test_generator_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(3)
        def failing_gen():
            yield 1
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == 2

        gen = failing_gen()
        assert next(gen) == 1

        with pytest.raises(ValueError, match="test error"):
            next(gen)

        # Should revert to 2 despite exception
        assert ctx.latency_sensitivity == 2

    def test_generator_function_early_exit(self):
        """Test that early exit from generator still pops stack."""

        @latency_sensitive(3)
        def gen_func():
            yield from range(10)

        ctx = Context.get()
        assert ctx.latency_sensitivity == 2

        # Only consume first 3 values
        gen = gen_func()
        results = [next(gen) for _ in range(3)]
        assert results == [0, 1, 2]

        # Close generator early
        gen.close()

        # Should still be able to access context after early exit
        # Note: Stack will pop when generator is garbage collected
        assert ctx.latency_sensitivity == 2


class TestDecoratorAsyncGeneratorFunctions:
    """Tests for @latency_sensitive decorator on async generator functions."""

    async def test_async_generator_function_with_int(self):
        """Test decorator on async generator function with integer value."""

        @latency_sensitive(3)
        async def async_gen_func():
            for i in range(3):
                yield (i, Context.get().latency_sensitivity)

        # Outside decorator, should be 2
        assert Context.get().latency_sensitivity == 2

        # Inside decorator, should be 3
        results = [item async for item in async_gen_func()]
        assert len(results) == 3
        for i, sensitivity in results:
            assert sensitivity == 3

        # After decorator, back to 2
        assert Context.get().latency_sensitivity == 2

    async def test_async_generator_function_with_lower_int(self):
        """Test decorator on async generator function with lower integer value."""

        @latency_sensitive(1)
        async def async_gen_func():
            for i in range(2):
                yield Context.get().latency_sensitivity

        results = [item async for item in async_gen_func()]
        # 1 is in stack, but default 2 has higher priority
        assert all(s == 2 for s in results)

    async def test_async_generator_function_priority_nesting(self):
        """Test priority-based nesting with async generator functions."""

        @latency_sensitive(1)
        async def low_async_gen():
            yield Context.get().latency_sensitivity

        @latency_sensitive(3)
        async def high_async_gen():
            # Get first value from low_async_gen while in 3 context
            async for val in low_async_gen():
                low_result = val
                break
            yield Context.get().latency_sensitivity, low_result

        async for outer, inner in high_async_gen():
            # Both should be 3 due to priority
            assert outer == 3
            assert inner == 3

    async def test_async_generator_function_yields_values(self):
        """Test that decorator preserves yielded values."""

        @latency_sensitive(3)
        async def async_gen_with_values(n):
            for i in range(n):
                yield i * 2

        results = [item async for item in async_gen_with_values(4)]
        assert results == [0, 2, 4, 6]

    async def test_async_generator_function_with_args_kwargs(self):
        """Test that decorator preserves arguments."""

        @latency_sensitive(3)
        async def async_gen_with_args(*args, **kwargs):
            yield args
            yield kwargs

        results = [item async for item in async_gen_with_args(1, 2, 3, x=4, y=5)]
        assert results[0] == (1, 2, 3)
        assert results[1] == {"x": 4, "y": 5}

    async def test_async_generator_function_exception_propagates(self):
        """Test that exceptions propagate and stack still pops."""

        @latency_sensitive(3)
        async def failing_async_gen():
            yield 1
            raise ValueError("test error")

        ctx = Context.get()
        assert ctx.latency_sensitivity == 2

        agen = failing_async_gen()
        assert await agen.__anext__() == 1

        with pytest.raises(ValueError, match="test error"):
            await agen.__anext__()

        # Should revert to 2 despite exception
        assert ctx.latency_sensitivity == 2

    async def test_async_generator_function_early_exit(self):
        """Test that early exit from async generator still pops stack."""

        @latency_sensitive(3)
        async def async_gen_func():
            for i in range(10):
                yield i

        ctx = Context.get()
        assert ctx.latency_sensitivity == 2

        # Only consume first 3 values
        agen = async_gen_func()
        results = []
        for _ in range(3):
            results.append(await agen.__anext__())
        assert results == [0, 1, 2]

        # Close async generator early
        await agen.aclose()

        # Should revert to 2 after close
        assert ctx.latency_sensitivity == 2

    async def test_mixed_async_and_async_gen_nesting(self):
        """Test that async functions and async generators can nest together."""

        @latency_sensitive(1)
        async def async_func():
            return Context.get().latency_sensitivity

        @latency_sensitive(3)
        async def high_async_gen():
            # 3 takes precedence
            async_result = await async_func()
            gen_result = Context.get().latency_sensitivity
            yield async_result, gen_result

        async for async_result, gen_result in high_async_gen():
            assert async_result == 3
            assert gen_result == 3
