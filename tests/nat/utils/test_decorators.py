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

import warnings

import pytest

from nat.utils.decorators import _warning_issued
from nat.utils.decorators import deprecated
from nat.utils.decorators import issue_deprecation_warning


# Reset warning state before each test
@pytest.fixture(autouse=True)
def clear_warnings():
    _warning_issued.clear()
    yield
    _warning_issued.clear()


def test_sync_function_logs_warning_once():
    """Test that a sync function logs deprecation warning only once."""
    @deprecated(removal_version="2.0.0", replacement="new_function")
    def sync_function():
        return "test"

    # First call should issue warning
    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in version 2.0.0. Use 'new_function' instead."):
        result = sync_function()

    assert result == "test"

    # Second call should not issue warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = sync_function()

    assert result == "test"


def test_async_function_logs_warning_once():
    """Test that an async function logs deprecation warning only once."""
    @deprecated(removal_version="2.0.0", replacement="new_async_function")
    async def async_function():
        return "async_test"

    async def run_test():
        # First call should issue warning
        with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in version 2.0.0. Use 'new_async_function' instead."):
            result = await async_function()

        assert result == "async_test"

        # Second call should not issue warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = await async_function()

        assert result == "async_test"

    import asyncio
    asyncio.run(run_test())


def test_generator_function_logs_warning_once():
    """Test that a generator function logs deprecation warning only once."""
    @deprecated(removal_version="2.0.0", replacement="new_generator")
    def generator_function():
        yield 1
        yield 2
        yield 3

    # First call should issue warning
    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in version 2.0.0. Use 'new_generator' instead."):
        gen = generator_function()
        results = list(gen)

    assert results == [1, 2, 3]

    # Second call should not issue warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gen = generator_function()
        results = list(gen)

    assert results == [1, 2, 3]


def test_async_generator_function_logs_warning_once():
    """Test that an async generator function logs deprecation warning only once."""
    @deprecated(removal_version="2.0.0", replacement="new_async_generator")
    async def async_generator_function():
        yield 1
        yield 2
        yield 3

    async def run_test():
        # First call should issue warning
        with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in version 2.0.0. Use 'new_async_generator' instead."):
            gen = async_generator_function()
            results = []
            async for item in gen:
                results.append(item)

        assert results == [1, 2, 3]

        # Second call should not issue warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gen = async_generator_function()
            results = []
            async for item in gen:
                results.append(item)

        assert results == [1, 2, 3]

    import asyncio
    asyncio.run(run_test())


def test_deprecation_with_feature_name():
    """Test deprecation warning with feature name."""
    @deprecated(feature_name="Old Feature", removal_version="2.0.0")
    def feature_function():
        return "test"

    with pytest.warns(DeprecationWarning, match="The Old Feature feature is deprecated and will be removed in version 2.0.0."):
        result = feature_function()

    assert result == "test"


def test_deprecation_with_reason():
    """Test deprecation warning with reason."""
    @deprecated(reason="This function has performance issues", replacement="fast_function")
    def slow_function():
        return "test"

    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in a future release. Reason: This function has performance issues. Use 'fast_function' instead."):
        result = slow_function()

    assert result == "test"


def test_deprecation_with_metadata():
    """Test deprecation warning with metadata."""
    @deprecated(metadata={"author": "test", "version": "1.0"})
    def metadata_function():
        return "test"

    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in a future release. Function: .* | Metadata: {'author': 'test', 'version': '1.0'}"):
        result = metadata_function()

    assert result == "test"


def test_deprecation_decorator_factory():
    """Test deprecation decorator factory usage."""
    @deprecated(removal_version="2.0.0", replacement="new_function")
    def factory_function():
        return "test"

    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in version 2.0.0. Use 'new_function' instead."):
        result = factory_function()

    assert result == "test"


def test_issue_deprecation_warning_directly():
    """Test calling issue_deprecation_warning directly."""
    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in a future release. Function: test_function"):
        issue_deprecation_warning("test_function")

    # Second call should not issue warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        issue_deprecation_warning("test_function")


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


def test_compatibility_alias():
    """Test that the compatibility alias works."""
    from nat.utils.decorators import aiq_deprecated

    @aiq_deprecated(removal_version="2.0.0")
    def alias_function():
        return "test"

    with pytest.warns(DeprecationWarning, match="This function is deprecated and will be removed in version 2.0.0."):
        result = alias_function()

    assert result == "test"