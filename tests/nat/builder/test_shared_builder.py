# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.builder.builder import Builder
from nat.builder.workflow_builder import SharedWorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.component import ComponentScope
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig


class SharedFunctionConfig(FunctionBaseConfig, name="test_shared_function"):
    """Test function with shared scope."""
    scope: ComponentScope = ComponentScope.SHARED


class PerUserFunctionConfig(FunctionBaseConfig, name="test_per_user_function"):
    """Test function with per-user scope."""
    scope: ComponentScope = ComponentScope.PER_USER


class SharedFunctionGroupConfig(FunctionGroupBaseConfig, name="test_shared_group"):
    """Test function group with shared scope."""
    scope: ComponentScope = ComponentScope.SHARED


class PerUserFunctionGroupConfig(FunctionGroupBaseConfig, name="test_per_user_group"):
    """Test function group with per-user scope."""
    scope: ComponentScope = ComponentScope.PER_USER


async def test_shared_builder_should_build_component():
    """Test SharedWorkflowBuilder only builds shared components."""
    builder = SharedWorkflowBuilder()

    # Should build shared components
    assert builder._should_build_component(SharedFunctionConfig()) is True
    assert builder._should_build_component(SharedFunctionGroupConfig()) is True

    # Should NOT build per-user components
    assert builder._should_build_component(PerUserFunctionConfig()) is False
    assert builder._should_build_component(PerUserFunctionGroupConfig()) is False


async def test_shared_builder_skips_auth_providers():
    """Test SharedWorkflowBuilder never builds auth providers."""
    from nat.data_models.authentication import AuthProviderBaseConfig

    class TestAuthConfig(AuthProviderBaseConfig, name="test_auth"):
        pass

    builder = SharedWorkflowBuilder()

    # Auth providers are always per-user
    assert builder._should_build_component(TestAuthConfig()) is False


async def test_shared_builder_populate_skips_workflow():
    """Test SharedWorkflowBuilder enforces skip_workflow=True."""
    config = Config()

    async with SharedWorkflowBuilder() as builder:
        # Should succeed with skip_workflow=True
        await builder.populate_builder(config, skip_workflow=True)

        # Should raise error with skip_workflow=False
        with pytest.raises(ValueError, match="Shared builder does not support populating with workflow"):
            await builder.populate_builder(config, skip_workflow=False)


async def test_shared_builder_cannot_build_workflow():
    """Test SharedWorkflowBuilder.build() raises error."""
    async with SharedWorkflowBuilder() as builder:
        with pytest.raises(RuntimeError, match="Shared builder does not support building workflows"):
            await builder.build()


async def test_shared_builder_validates_dependencies():
    """Test SharedWorkflowBuilder validates no shared→per-user dependencies."""
    from nat.cli.type_registry import GlobalTypeRegistry

    # Register test functions
    registry = GlobalTypeRegistry.get()

    @register_function(config_type=SharedFunctionConfig)
    async def shared_fn(config: SharedFunctionConfig, builder: Builder):
        # Shared function trying to depend on per-user function
        per_user_fn = await builder.get_function("test_per_user_function")  # Invalid dependency!

        async def _inner(x: str) -> str:
            return x

        yield _inner

    @register_function(config_type=PerUserFunctionConfig)
    async def per_user_fn(config: PerUserFunctionConfig, builder: Builder):

        async def _inner(x: str) -> str:
            return x

        yield _inner

    # Create config with invalid dependency
    config = Config()
    config.functions = {
        "test_shared_function": SharedFunctionConfig(), "test_per_user_function": PerUserFunctionConfig()
    }

    # Should raise ValueError indicating function not found (because SharedBuilder doesn't build per-user)
    async with SharedWorkflowBuilder(registry=registry) as builder:
        with pytest.raises(ValueError, match="Function.*not found"):
            await builder.populate_builder(config, skip_workflow=True)
