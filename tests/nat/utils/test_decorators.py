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

from nat.utils.decorators import _warning_issued
from nat.utils.decorators import deprecated
from nat.utils.decorators import issue_deprecation_warning


# Reset warning state before each test
@pytest.fixture(name="clear_warnings", autouse=True)
def fixture_clear_warnings():
    _warning_issued.clear()
    yield
    _warning_issued.clear()


def test_sync_function_logs_warning_once(caplog):
    """Test that a sync function logs deprecation warning only once."""
    caplog.set_level(logging.WARNING)

    @deprecated(removal_version="2.0.0", replacement="new_function")
    def sync_function():
        return "test"

    # First call should issue warning
    result = sync_function()
    assert result == "test"
    old_fn = "test_decorators.test_sync_function_logs_warning_once.<locals>.sync_function"
    new_fn = "new_function"
    expected = f"Function {old_fn} is deprecated and will be removed in version 2.0.0. Use '{new_fn}' instead."
    assert any(expected in rec.getMessage() for rec in caplog.records)

    caplog.clear()

    # Second call should not issue warning
    result = sync_function()
    assert result == "test"
    assert not caplog.records


def test_async_function_logs_warning_once(caplog):
    """Test that an async function logs deprecation warning only once."""
    caplog.set_level(logging.WARNING)

    @deprecated(removal_version="2.0.0", replacement="new_async_function")
    async def async_function():
        return "async_test"

    async def run_test():
        # First call should issue warning
        result = await async_function()
        assert result == "async_test"
        old_fn = "test_decorators.test_async_function_logs_warning_once.<locals>.async_function"
        new_fn = "new_async_function"
        expected = f"Function {old_fn} is deprecated and will be removed in version 2.0.0. Use '{new_fn}' instead."
        assert any(expected in rec.getMessage() for rec in caplog.records)

        caplog.clear()

        # Second call should not issue warning
        result = await async_function()
        assert result == "async_test"
        assert not caplog.records

    import asyncio
    asyncio.run(run_test())


def test_generator_function_logs_warning_once(caplog):
    """Test that a generator function logs deprecation warning only once."""
    caplog.set_level(logging.WARNING)

    @deprecated(removal_version="2.0.0", replacement="new_generator")
    def generator_function():
        yield 1
        yield 2
        yield 3

    # First call should issue warning
    gen = generator_function()
    results = list(gen)
    assert results == [1, 2, 3]
    old_fn = "test_decorators.test_generator_function_logs_warning_once.<locals>.generator_function"
    new_fn = "new_generator"
    expected = f"Function {old_fn} is deprecated and will be removed in version 2.0.0. Use '{new_fn}' instead."
    assert any(expected in rec.getMessage() for rec in caplog.records)

    caplog.clear()

    # Second call should not issue warning
    gen = generator_function()
    results = list(gen)
    assert results == [1, 2, 3]
    assert not caplog.records


def test_async_generator_function_logs_warning_once(caplog):
    """Test that an async generator function logs deprecation warning only once."""
    caplog.set_level(logging.WARNING)

    @deprecated(removal_version="2.0.0", replacement="new_async_generator")
    async def async_generator_function():
        yield 1
        yield 2
        yield 3

    async def run_test():
        # First call should issue warning
        gen = async_generator_function()
        results = []
        async for item in gen:
            results.append(item)

        assert results == [1, 2, 3]
        old_fn = "test_decorators.test_async_generator_function_logs_warning_once.<locals>.async_generator_function"
        new_fn = "new_async_generator"
        expected = f"Function {old_fn} is deprecated and will be removed in version 2.0.0. Use '{new_fn}' instead."
        assert any(expected in rec.getMessage() for rec in caplog.records)

        caplog.clear()

        # Second call should not issue warning
        gen = async_generator_function()
        results = []
        async for item in gen:
            results.append(item)

        assert results == [1, 2, 3]
        assert not caplog.records

    import asyncio
    asyncio.run(run_test())


def test_deprecation_with_feature_name(caplog):
    """Test deprecation warning with feature name."""
    caplog.set_level(logging.WARNING)

    @deprecated(feature_name="Old Feature", removal_version="2.0.0")
    def feature_function():
        return "test"

    result = feature_function()
    assert result == "test"
    assert any("Old Feature is deprecated and will be removed in version 2.0.0." in rec.getMessage()
               for rec in caplog.records)


def test_deprecation_with_reason(caplog):
    """Test deprecation warning with reason."""
    caplog.set_level(logging.WARNING)

    @deprecated(reason="This function has performance issues", replacement="fast_function")
    def slow_function():
        return "test"

    result = slow_function()
    assert result == "test"
    old_fn = "test_decorators.test_deprecation_with_reason.<locals>.slow_function"
    new_fn = "fast_function"
    expected = (f"Function {old_fn} is deprecated and will be removed in a future release. "
                f"Reason: This function has performance issues. Use '{new_fn}' instead.")
    assert any(expected in rec.getMessage() for rec in caplog.records)


def test_deprecation_with_metadata(caplog):
    """Test deprecation warning with metadata."""
    caplog.set_level(logging.WARNING)

    @deprecated(metadata={"author": "test", "version": "1.0"})
    def metadata_function():
        return "test"

    result = metadata_function()
    assert result == "test"
    old_fn = "test_decorators.test_deprecation_with_metadata.<locals>.metadata_function"
    expected = (f"Function {old_fn} is deprecated and will be removed in a future release. "
                "| Metadata: {'author': 'test', 'version': '1.0'}")
    assert any(expected in rec.getMessage() for rec in caplog.records)


def test_deprecation_decorator_factory(caplog):
    """Test deprecation decorator factory usage."""
    caplog.set_level(logging.WARNING)

    @deprecated(removal_version="2.0.0", replacement="new_function")
    def factory_function():
        return "test"

    result = factory_function()
    assert result == "test"
    old_fn = "test_decorators.test_deprecation_decorator_factory.<locals>.factory_function"
    new_fn = "new_function"
    expected = f"Function {old_fn} is deprecated and will be removed in version 2.0.0. Use '{new_fn}' instead."
    assert any(expected in rec.getMessage() for rec in caplog.records)


def test_issue_deprecation_warning_directly(caplog):
    """Test calling issue_deprecation_warning directly."""
    caplog.set_level(logging.WARNING)

    issue_deprecation_warning("test_function")
    assert any("Function test_function is deprecated and will be removed in a future release." in rec.getMessage()
               for rec in caplog.records)

    caplog.clear()

    # Second call should not issue warning
    issue_deprecation_warning("test_function")
    assert not caplog.records


def test_metadata_validation():
    """Test that metadata validation works correctly."""
    with pytest.raises(TypeError, match="metadata must be a dict"):

        @deprecated(metadata="not-a-dict")
        def invalid_metadata_function():
            pass

    with pytest.raises(TypeError, match="All metadata keys must be strings"):

        @deprecated(metadata={1: "value"})
        def invalid_key_function():
            pass
