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

from nat.builder.context import Context
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import SharedWorkflowBuilder
from nat.builder.workflow_builder import UserWorkflowBuilder
from nat.data_models.config import Config

from .conftest import TAuthConfig
from .conftest import TEmbedderProviderConfig
from .conftest import TLLMProviderConfig
from .conftest import TMemoryConfig
from .conftest import TObjectStoreConfig
from .conftest import TPerUserFunctionConfig
from .conftest import TRetrieverProviderConfig
from .conftest import TSharedFunctionConfig
from .conftest import TTTCStrategyConfig


async def test_user_builder_has_user_id():
    """Test UserWorkflowBuilder has user_id."""
    async with SharedWorkflowBuilder() as shared_builder:
        user_builder = UserWorkflowBuilder(shared_builder=shared_builder, user_id="test_user")
        assert user_builder.user_id == "test_user"


async def test_user_builder_references_shared_components():
    """Test UserWorkflowBuilder references shared resources without rebuilding them."""
    # Create config with mock components (matching pattern from test_builder.py)
    config = Config(
        llms={"test_llm": TLLMProviderConfig()},
        embedders={"test_embedder": TEmbedderProviderConfig()},
        memory={"test_memory": TMemoryConfig()},
        object_stores={"test_object_store": TObjectStoreConfig()},
        retrievers={"test_retriever": TRetrieverProviderConfig()},
        ttc_strategies={"test_ttc_strategy": TTTCStrategyConfig()},
    )

    async with SharedWorkflowBuilder() as shared_builder:
        await shared_builder.populate_builder(config, skip_workflow=True)
        user_builder = UserWorkflowBuilder(user_id="test_user", shared_builder=shared_builder)

        # Should reference same instances, not copies
        assert user_builder._llms is shared_builder._llms
        assert user_builder._embedders is shared_builder._embedders
        assert user_builder._memory_clients is shared_builder._memory_clients
        assert user_builder._object_stores is shared_builder._object_stores
        assert user_builder._retrievers is shared_builder._retrievers
        assert user_builder._ttc_strategies is shared_builder._ttc_strategies

        # Should copy function groups/functions, not reference them
        assert user_builder._function_groups is not shared_builder._function_groups
        assert user_builder._functions is not shared_builder._functions
        assert user_builder.function_group_dependencies is not shared_builder.function_group_dependencies
        assert user_builder.function_dependencies is not shared_builder.function_dependencies


async def test_user_builder_sets_user_id_in_context():
    """Test UserWorkflowBuilder sets user_id in ContextState on enter."""
    async with SharedWorkflowBuilder() as shared_builder:
        # user_id should not be set yet
        context_user_id = Context.get().user_id
        assert context_user_id is None

        # Enter user builder context
        async with UserWorkflowBuilder(user_id="test_user_123", shared_builder=shared_builder) as user_builder:
            # user_id should be set in context
            context_user_id = Context.get().user_id
            assert context_user_id == "test_user_123"

        # After exit, user_id should be reset
        context_user_id = Context.get().user_id
        assert context_user_id is None


async def test_user_builder_should_build_component():
    """Test UserWorkflowBuilder only builds per-user components."""

    async with SharedWorkflowBuilder() as shared_builder:
        user_builder = UserWorkflowBuilder(user_id="test", shared_builder=shared_builder)
        assert user_builder._should_build_component(TPerUserFunctionConfig()) is True
        assert user_builder._should_build_component(TAuthConfig()) is True
        assert user_builder._should_build_component(TSharedFunctionConfig()) is False


async def test_user_builder_builds_workflow():
    """Test UserWorkflowBuilder can build workflows"""
    # Create config with both shared and per-user components
    config = Config(
        functions={
            "test_per_user_function": TPerUserFunctionConfig(),
            "test_shared_function": TSharedFunctionConfig(),
        },
        authentication={"test_auth": TAuthConfig()},
    )

    async with SharedWorkflowBuilder() as shared_builder:
        await shared_builder.populate_builder(config)

        async with UserWorkflowBuilder(user_id="test_user", shared_builder=shared_builder) as user_builder:
            await user_builder.populate_builder(config)
            workflow = await user_builder.build()
            assert isinstance(workflow, Workflow)
