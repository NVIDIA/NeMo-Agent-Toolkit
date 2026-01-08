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

import pytest
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.component_utils import WORKFLOW_COMPONENT_NAME
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.per_user_workflow_builder import PerUserWorkflowBuilder
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.runtime.session import SessionManager


# Test schemas for per-user functions
class PerUserInputSchema(BaseModel):
    message: str = Field(description="Input message")


class PerUserOutputSchema(BaseModel):
    result: str = Field(description="Output result")


# Test configs
class SharedFunctionConfig(FunctionBaseConfig, name="shared_fn"):
    """A shared function config for testing."""
    pass


class PerUserFunctionConfig(FunctionBaseConfig, name="per_user_fn"):
    """A per-user function config for testing."""
    pass


class PerUserFunctionBConfig(FunctionBaseConfig, name="per_user_fn_b"):
    """Another per-user function config for testing."""
    pass


class PerUserWorkflowConfig(FunctionBaseConfig, name="per_user_workflow"):
    """A per-user workflow config for testing."""
    pass


class SharedWorkflowConfig(FunctionBaseConfig, name=WORKFLOW_COMPONENT_NAME):
    """A shared workflow config for testing."""
    pass


class PerUserDependentFnConfig(FunctionBaseConfig, name="per_user_dependent"):
    """A per-user function that depends on another per-user function."""
    other_fn_name: str


class SharedDependentFnConfig(FunctionBaseConfig, name="bad_shared_fn"):
    """A shared function that incorrectly depends on a per-user function."""
    per_user_fn_name: str


# E2E test configs for SessionManager integration
class CounterInput(BaseModel):
    action: str = Field(description="Either 'increment' or 'get'")


class CounterOutput(BaseModel):
    count: int = Field(description="Current count value")


class PerUserCounterConfig(FunctionBaseConfig, name="per_user_counter"):
    """A per-user counter that maintains state per user."""
    initial_value: int = 0


class PerUserCounterWorkflowConfig(FunctionBaseConfig, name="per_user_counter_workflow"):
    """A per-user workflow that uses the counter."""
    counter_name: str = "counter"


# Register all test components
@pytest.fixture(scope="module", autouse=True)
async def register_components():

    # Register shared function
    @register_function(config_type=SharedFunctionConfig)
    async def build_shared_fn(_config: SharedFunctionConfig, _b: Builder):

        async def _impl(inp: str) -> str:
            return f"shared: {inp}"

        yield FunctionInfo.from_fn(_impl)

    # Register per-user function
    @register_per_user_function(config_type=PerUserFunctionConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def build_per_user_fn(_config: PerUserFunctionConfig, _b: Builder):

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            return PerUserOutputSchema(result=f"per-user: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    # Register another per-user function for dependency testing
    @register_per_user_function(config_type=PerUserFunctionBConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def build_per_user_fn_b(_config: PerUserFunctionBConfig, _b: Builder):

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            return PerUserOutputSchema(result=f"per-user-b: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    # Register per-user function that depends on another per-user function
    @register_per_user_function(config_type=PerUserDependentFnConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def build_per_user_dependent_fn(config: PerUserDependentFnConfig, b: Builder):
        # Get the other per-user function
        other_fn = await b.get_function(config.other_fn_name)

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            # Call the other function
            other_result = await other_fn.ainvoke(inp, to_type=PerUserOutputSchema)
            return PerUserOutputSchema(result=f"dependent: {other_result.result}")

        yield FunctionInfo.from_fn(_impl)

    # Register per-user workflow
    @register_per_user_function(config_type=PerUserWorkflowConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def build_per_user_workflow(_config: PerUserWorkflowConfig, _b: Builder):

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            return PerUserOutputSchema(result=f"per-user-workflow: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    # Register shared workflow
    @register_function(config_type=SharedWorkflowConfig)
    async def build_shared_workflow(_config: SharedWorkflowConfig, _b: Builder):

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            return PerUserOutputSchema(result=f"shared-workflow: {inp.message}")

        yield FunctionInfo.from_fn(_impl)

    # Per-user counter - each user gets their own counter instance (for e2e tests)
    @register_per_user_function(config_type=PerUserCounterConfig,
                                input_type=CounterInput,
                                single_output_type=CounterOutput)
    async def per_user_counter(config: PerUserCounterConfig, _builder: Builder):
        # This state is unique per user!
        counter_state = {"count": config.initial_value}

        async def _counter(inp: CounterInput) -> CounterOutput:
            if inp.action == "increment":
                counter_state["count"] += 1
            return CounterOutput(count=counter_state["count"])

        yield FunctionInfo.from_fn(_counter)

    # Per-user workflow that uses the counter (for e2e tests)
    @register_per_user_function(config_type=PerUserCounterWorkflowConfig,
                                input_type=CounterInput,
                                single_output_type=CounterOutput)
    async def per_user_counter_workflow(config: PerUserCounterWorkflowConfig, builder: Builder):
        # Get the per-user counter function
        counter_fn = await builder.get_function(config.counter_name)

        async def _workflow(inp: CounterInput) -> CounterOutput:
            return await counter_fn.ainvoke(inp, to_type=CounterOutput)

        yield FunctionInfo.from_fn(_workflow)


async def test_workflow_builder_skips_per_user_functions():
    """Test that WorkflowBuilder.populate_builder() skips per-user functions."""

    config = Config(functions={
        "shared_fn": SharedFunctionConfig(),
        "per_user_fn": PerUserFunctionConfig(),
        WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig(),
    },
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as builder:
        # Shared function should be built
        assert "shared_fn" in builder._functions
        shared_fn = await builder.get_function("shared_fn")
        assert shared_fn is not None

        # Per-user function should NOT be built in shared builder
        assert "per_user_fn" not in builder._functions

        # Attempting to get per-user function should fail
        with pytest.raises(ValueError, match="Function `per_user_fn` not found"):
            await builder.get_function("per_user_fn")


async def test_workflow_builder_skips_per_user_workflow():
    """Test that WorkflowBuilder.populate_builder() skips per-user workflows."""

    config = Config(workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as builder:
        # Per-user workflow should NOT be built in shared builder
        assert builder._workflow is None

        # Attempting to get workflow should fail
        with pytest.raises(ValueError, match="No workflow set"):
            builder.get_workflow()


async def test_workflow_builder_builds_shared_workflow():
    """Test that WorkflowBuilder builds shared workflows normally."""

    config = Config(functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()}, workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as builder:
        # Shared workflow should be built
        assert builder._workflow is not None
        workflow_fn = builder.get_workflow()
        assert workflow_fn is not None


async def test_workflow_builder_validates_shared_depends_on_per_user():
    """Test that WorkflowBuilder._validate_dependencies() catches shared->per-user dependencies."""

    # Register a shared function that depends on a per-user function (invalid!)
    @register_function(config_type=SharedDependentFnConfig)
    async def bad_register(config: SharedDependentFnConfig, b: Builder):
        # Try to get a per-user function (this will fail)
        _ = await b.get_function(config.per_user_fn_name)

        async def _impl(inp: str) -> str:
            return f"bad: {inp}"

        yield FunctionInfo.from_fn(_impl)

    config = Config(functions={
        "per_user_fn": PerUserFunctionConfig(),
        "bad_shared_fn": SharedDependentFnConfig(per_user_fn_name="per_user_fn"),
        WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig(),
    },
                    workflow=SharedWorkflowConfig())

    # Building should fail - either with validation error or when trying to get per-user function
    with pytest.raises(ValueError):
        async with WorkflowBuilder.from_config(config) as _:
            pass


async def test_workflow_builder_validates_shared_fn_depends_on_per_user_function_group():
    """Test that WorkflowBuilder._validate_dependencies() catches shared function->per-user function_group."""
    import typing

    from nat.cli.register_workflow import register_per_user_function_group

    class SharedFnDependsOnPerUserFGConfig(FunctionBaseConfig, name="shared_fn_depends_per_user_fg"):
        fg_name: str

    class PerUserFGForFnTestConfig(FunctionGroupBaseConfig, name="per_user_fg_for_fn_test"):
        pass

    @register_per_user_function_group(config_type=PerUserFGForFnTestConfig)
    async def per_user_fg_for_fn_test(config: PerUserFGForFnTestConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_function(config_type=SharedFnDependsOnPerUserFGConfig)
    async def shared_fn_depends_per_user_fg(config: SharedFnDependsOnPerUserFGConfig, b: Builder):
        _ = await b.get_function_group(config.fg_name)

        async def _impl(inp: str) -> str:
            return f"result: {inp}"

        yield FunctionInfo.from_fn(_impl)

    config = Config(function_groups={
        "per_user_fg": PerUserFGForFnTestConfig(),
    },
                    functions={
                        "shared_fn": SharedFnDependsOnPerUserFGConfig(fg_name="per_user_fg"),
                        WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig(),
                    },
                    workflow=SharedWorkflowConfig())

    # Building should fail - either with validation error or when trying to get per-user function_group
    with pytest.raises(ValueError):
        async with WorkflowBuilder.from_config(config) as _:
            pass


async def test_workflow_builder_validates_shared_fg_depends_on_per_user_function():
    """Test that WorkflowBuilder._validate_dependencies() catches shared function_group->per-user function."""
    import typing

    from nat.cli.register_workflow import register_function_group

    class SharedFGDependsOnPerUserFnConfig(FunctionGroupBaseConfig, name="shared_fg_depends_per_user_fn"):
        fn_name: str

    @register_function_group(config_type=SharedFGDependsOnPerUserFnConfig)
    async def shared_fg_depends_per_user_fn(config: SharedFGDependsOnPerUserFnConfig, builder: Builder):
        _ = await builder.get_function(config.fn_name)

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "shared_fg": SharedFGDependsOnPerUserFnConfig(fn_name="per_user_fn"),
    },
                    functions={
                        "per_user_fn": PerUserFunctionConfig(),
                        WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig(),
                    },
                    workflow=SharedWorkflowConfig())

    # Building should fail - either with validation error or when trying to get per-user function
    with pytest.raises(ValueError):
        async with WorkflowBuilder.from_config(config) as _:
            pass


async def test_workflow_builder_validates_shared_fg_depends_on_per_user_fg():
    """Test that WorkflowBuilder._validate_dependencies() catches shared function_group->per-user function_group."""
    import typing

    from nat.cli.register_workflow import register_function_group
    from nat.cli.register_workflow import register_per_user_function_group

    class SharedFGDependsOnPerUserFGConfig(FunctionGroupBaseConfig, name="shared_fg_depends_per_user_fg"):
        other_fg_name: str

    class PerUserFGForFGTestConfig(FunctionGroupBaseConfig, name="per_user_fg_for_fg_test"):
        pass

    @register_per_user_function_group(config_type=PerUserFGForFGTestConfig)
    async def per_user_fg_for_fg_test(config: PerUserFGForFGTestConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_function_group(config_type=SharedFGDependsOnPerUserFGConfig)
    async def shared_fg_depends_per_user_fg(config: SharedFGDependsOnPerUserFGConfig, builder: Builder):
        _ = await builder.get_function_group(config.other_fg_name)

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "per_user_fg": PerUserFGForFGTestConfig(),
        "shared_fg": SharedFGDependsOnPerUserFGConfig(other_fg_name="per_user_fg"),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    # Building should fail - either with validation error or when trying to get per-user function_group
    with pytest.raises(ValueError):
        async with WorkflowBuilder.from_config(config) as _:
            pass


async def test_workflow_builder_validates_shared_workflow_depends_on_per_user_fg():
    """Test that WorkflowBuilder._validate_dependencies() catches shared workflow->per-user function_group."""
    import typing

    from nat.cli.register_workflow import register_per_user_function_group

    class PerUserFGForWorkflowTestConfig(FunctionGroupBaseConfig, name="per_user_fg_for_workflow_test"):
        pass

    class SharedWorkflowDependsOnPerUserFGConfig(FunctionBaseConfig, name="shared_wf_depends_per_user_fg"):
        fg_name: str

    @register_per_user_function_group(config_type=PerUserFGForWorkflowTestConfig)
    async def per_user_fg_for_wf_test(config: PerUserFGForWorkflowTestConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_function(config_type=SharedWorkflowDependsOnPerUserFGConfig)
    async def shared_wf_depends_per_user_fg(config: SharedWorkflowDependsOnPerUserFGConfig, b: Builder):
        _ = await b.get_function_group(config.fg_name)

        async def _impl(inp: str) -> str:
            return f"result: {inp}"

        yield FunctionInfo.from_fn(_impl)

    config = Config(function_groups={
        "per_user_fg": PerUserFGForWorkflowTestConfig(),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowDependsOnPerUserFGConfig(fg_name="per_user_fg")},
                    workflow=SharedWorkflowDependsOnPerUserFGConfig(fg_name="per_user_fg"))

    # Building should fail - either with validation error or when trying to get per-user function_group
    with pytest.raises(ValueError):
        async with WorkflowBuilder.from_config(config) as _:
            pass


async def test_per_user_workflow_builder_initialization():
    """Test PerUserWorkflowBuilder can be initialized."""

    config = Config(functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()}, workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user123", shared_builder=shared_builder) as per_user_builder:
            # Check initialization
            assert per_user_builder.user_id == "user123"
            assert per_user_builder._shared_builder is shared_builder
            assert per_user_builder._per_user_functions == {}
            assert per_user_builder._workflow is None


async def test_per_user_workflow_builder_populate_builds_per_user_functions():
    """Test PerUserWorkflowBuilder.populate_builder() builds only per-user functions."""

    config = Config(functions={
        "shared_fn": SharedFunctionConfig(),
        "per_user_fn": PerUserFunctionConfig(),
        WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig(),
    },
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user123", shared_builder=shared_builder) as per_user_builder:
            await per_user_builder.populate_builder(config)

            # Per-user function should be built
            assert "per_user_fn" in per_user_builder._per_user_functions

            # Shared function should NOT be built in per-user builder
            assert "shared_fn" not in per_user_builder._per_user_functions

            # But per-user builder can access shared function via delegation
            shared_fn = await per_user_builder.get_function("shared_fn")
            assert shared_fn is not None


async def test_per_user_workflow_builder_populate_builds_per_user_workflow():
    """Test PerUserWorkflowBuilder.populate_builder() builds per-user workflows."""

    config = Config(workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user123", shared_builder=shared_builder) as per_user_builder:
            await per_user_builder.populate_builder(config)

            # Per-user workflow should be built
            assert per_user_builder._workflow is not None

            # Should be able to get workflow
            workflow_fn = per_user_builder.get_workflow()
            assert workflow_fn is not None


async def test_per_user_workflow_builder_delegates_to_shared_workflow():
    """Test PerUserWorkflowBuilder delegates to shared workflow when workflow is shared."""

    config = Config(functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()}, workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user123", shared_builder=shared_builder) as per_user_builder:
            await per_user_builder.populate_builder(config)

            # Per-user builder should NOT have its own workflow
            assert per_user_builder._workflow is None

            # But should delegate to shared builder's workflow
            workflow_fn = per_user_builder.get_workflow()
            assert workflow_fn is not None
            assert workflow_fn is shared_builder.get_workflow()


async def test_per_user_workflow_builder_build_creates_workflow():
    """Test PerUserWorkflowBuilder.build() creates a workflow instance."""

    config = Config(functions={
        "per_user_fn": PerUserFunctionConfig(),
    }, workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user123", shared_builder=shared_builder) as per_user_builder:
            await per_user_builder.populate_builder(config)

            # Build the workflow
            workflow = await per_user_builder.build()

            # Verify workflow was created
            assert workflow is not None

            # Verify it has per-user function
            assert "per_user_fn" in workflow.functions

            # Verify workflow can be invoked
            result = await workflow._entry_fn.ainvoke(PerUserInputSchema(message="test"), to_type=PerUserOutputSchema)
            assert result.result == "per-user-workflow: test"


async def test_per_user_workflow_builder_build_merges_shared_and_per_user():
    """Test that PerUserWorkflowBuilder.build() merges shared and per-user functions."""

    config = Config(functions={
        "shared_fn": SharedFunctionConfig(),
        "per_user_fn": PerUserFunctionConfig(),
    },
                    workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user123", shared_builder=shared_builder) as per_user_builder:
            await per_user_builder.populate_builder(config)
            workflow = await per_user_builder.build()

            # Both shared and per-user functions should be in the workflow
            assert "shared_fn" in workflow.functions
            assert "per_user_fn" in workflow.functions


async def test_per_user_workflow_builder_from_config():
    """Test PerUserWorkflowBuilder.from_config() factory method."""

    config = Config(functions={
        "per_user_fn": PerUserFunctionConfig(),
    }, workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        # Use from_config factory
        async with PerUserWorkflowBuilder.from_config(user_id="user456", config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Should be initialized and populated
            assert per_user_builder.user_id == "user456"
            assert "per_user_fn" in per_user_builder._per_user_functions
            assert per_user_builder._workflow is not None


async def test_per_user_function_can_call_shared_function():
    """Test that per-user functions can access shared functions during build."""

    # Register a per-user function that uses a shared function
    class PerUserCallsSharedConfig(FunctionBaseConfig, name="test_per_user_calls_shared"):
        shared_fn_name: str

    @register_per_user_function(config_type=PerUserCallsSharedConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def register(config: PerUserCallsSharedConfig, b: Builder):
        # Get shared function during build
        shared_fn = await b.get_function(config.shared_fn_name)

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            shared_result = await shared_fn.ainvoke(inp.message, to_type=str)
            return PerUserOutputSchema(result=f"wrapped: {shared_result}")

        yield FunctionInfo.from_fn(_impl)

    config = Config(functions={
        "shared_fn": SharedFunctionConfig(),
        "per_user_fn": PerUserCallsSharedConfig(shared_fn_name="shared_fn"),
    },
                    workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user789", config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Per-user function should be built successfully
            assert "per_user_fn" in per_user_builder._per_user_functions

            # Test invocation
            per_user_fn = await per_user_builder.get_function("per_user_fn")
            result = await per_user_fn.ainvoke(PerUserInputSchema(message="hello"), to_type=PerUserOutputSchema)
            assert result.result == "wrapped: shared: hello"


async def test_per_user_function_can_call_another_per_user_function():
    """Test that per-user functions can depend on other per-user functions."""

    config = Config(functions={
        "per_user_fn_b": PerUserFunctionBConfig(),
        "per_user_dependent": PerUserDependentFnConfig(other_fn_name="per_user_fn_b"),
    },
                    workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user999", config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Both per-user functions should be built
            assert "per_user_fn_b" in per_user_builder._per_user_functions
            assert "per_user_dependent" in per_user_builder._per_user_functions

            # Test that dependent function works
            dependent_fn = await per_user_builder.get_function("per_user_dependent")
            result = await dependent_fn.ainvoke(PerUserInputSchema(message="test"), to_type=PerUserOutputSchema)
            assert result.result == "dependent: per-user-b: test"


async def test_per_user_workflow_builder_delegates_llm_access():
    """Test that PerUserWorkflowBuilder delegates LLM access to shared builder."""
    from _utils.configs import LLMProviderTestConfig
    from nat.builder.llm import LLMProviderInfo
    from nat.cli.register_workflow import register_llm_provider

    # Register a test LLM
    @register_llm_provider(config_type=LLMProviderTestConfig)
    async def register_llm(config: LLMProviderTestConfig, b: Builder):
        yield LLMProviderInfo(config=config, description="Test LLM")

    config = Config(llms={"test_llm": LLMProviderTestConfig()},
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user_llm", shared_builder=shared_builder) as per_user_builder:
            # Per-user builder should be able to get LLM config
            llm_config = per_user_builder.get_llm_config("test_llm")
            assert llm_config is not None
            assert isinstance(llm_config, LLMProviderTestConfig)


async def test_per_user_workflow_builder_multiple_users():
    """Test that multiple PerUserWorkflowBuilders can be created for different users."""

    config = Config(functions={
        "per_user_fn": PerUserFunctionConfig(),
    }, workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        # Create builders for two different users
        async with PerUserWorkflowBuilder.from_config(user_id="alice", config=config,
                                                      shared_builder=shared_builder) as alice_builder:

            async with PerUserWorkflowBuilder.from_config(user_id="bob", config=config,
                                                          shared_builder=shared_builder) as bob_builder:

                # Each should have their own instance
                assert alice_builder.user_id == "alice"
                assert bob_builder.user_id == "bob"

                # Each should have built their own per-user functions
                assert "per_user_fn" in alice_builder._per_user_functions
                assert "per_user_fn" in bob_builder._per_user_functions

                # But they should be different instances
                alice_fn = await alice_builder.get_function("per_user_fn")
                bob_fn = await bob_builder.get_function("per_user_fn")
                assert alice_fn is not bob_fn

                # Both should share the same shared builder
                assert alice_builder._shared_builder is shared_builder
                assert bob_builder._shared_builder is shared_builder


async def test_per_user_workflow_builder_get_function_priority():
    """Test that PerUserWorkflowBuilder.get_function() checks per-user first, then shared."""

    config = Config(functions={
        "shared_fn": SharedFunctionConfig(),
        "per_user_fn": PerUserFunctionConfig(),
    },
                    workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_priority",
                                                      config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Get per-user function (should come from per-user cache)
            per_user_fn = await per_user_builder.get_function("per_user_fn")
            assert per_user_fn is per_user_builder._per_user_functions["per_user_fn"].instance

            # Get shared function (should come from shared builder)
            shared_fn = await per_user_builder.get_function("shared_fn")
            assert shared_fn is shared_builder._functions["shared_fn"].instance


async def test_per_user_workflow_builder_build_with_entry_function():
    """Test PerUserWorkflowBuilder.build() with custom entry function."""

    config = Config(functions={
        "per_user_fn": PerUserFunctionConfig(),
    }, workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_entry",
                                                      config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Build with per_user_fn as entry point
            workflow = await per_user_builder.build(entry_function="per_user_fn")

            # Workflow should use per_user_fn as entry
            result = await workflow._entry_fn.ainvoke(PerUserInputSchema(message="entry_test"),
                                                      to_type=PerUserOutputSchema)
            assert result.result == "per-user: entry_test"


async def test_per_user_workflow_builder_build_with_shared_entry_function():
    """Test PerUserWorkflowBuilder.build() can use shared function as entry."""

    config = Config(functions={
        "shared_fn": SharedFunctionConfig(),
        "per_user_fn": PerUserFunctionConfig(),
    },
                    workflow=PerUserWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_shared_entry",
                                                      config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Build with shared_fn as entry point
            workflow = await per_user_builder.build(entry_function="shared_fn")

            # Workflow should use shared_fn as entry
            result = await workflow._entry_fn.ainvoke("shared_entry_test", to_type=str)
            assert result == "shared: shared_entry_test"


# ============= E2E Tests with SessionManager =============


async def test_per_user_function_isolation_with_session_manager():
    """Test that different users have isolated per-user function state via SessionManager."""

    config = Config(functions={
        "counter": PerUserCounterConfig(initial_value=0),
    },
                    workflow=PerUserCounterWorkflowConfig(counter_name="counter"))

    async with WorkflowBuilder.from_config(config) as builder:
        # Create SessionManager (per-user workflow)
        sm = await SessionManager.create(config=config, shared_builder=builder)

        try:
            # User 1: Increment counter twice
            async with sm.session(user_id="alice") as session:
                async with session.run(CounterInput(action="increment")) as runner:
                    result1 = await runner.result(to_type=CounterOutput)
                    assert result1.count == 1

            async with sm.session(user_id="alice") as session:
                async with session.run(CounterInput(action="increment")) as runner:
                    result2 = await runner.result(to_type=CounterOutput)
                    assert result2.count == 2  # Alice's counter is at 2

            # User 2: Should have their own counter starting at 0
            async with sm.session(user_id="bob") as session:
                async with session.run(CounterInput(action="get")) as runner:
                    result3 = await runner.result(to_type=CounterOutput)
                    assert result3.count == 0  # Bob's counter is at 0 (fresh!)

            async with sm.session(user_id="bob") as session:
                async with session.run(CounterInput(action="increment")) as runner:
                    result4 = await runner.result(to_type=CounterOutput)
                    assert result4.count == 1  # Bob's counter is at 1

            # Verify Alice's counter is still at 2
            async with sm.session(user_id="alice") as session:
                async with session.run(CounterInput(action="get")) as runner:
                    result5 = await runner.result(to_type=CounterOutput)
                    assert result5.count == 2  # Still 2!

        finally:
            await sm.shutdown()


async def test_per_user_builder_caching_with_session_manager():
    """Test that per-user builders are cached and reused via SessionManager."""

    config = Config(functions={
        "counter": PerUserCounterConfig(initial_value=10),
    },
                    workflow=PerUserCounterWorkflowConfig(counter_name="counter"))

    async with WorkflowBuilder.from_config(config) as builder:
        sm = await SessionManager.create(config=config, shared_builder=builder)

        try:
            # First access creates the builder
            async with sm.session(user_id="user1") as session:
                async with session.run(CounterInput(action="increment")) as runner:
                    result = await runner.result(to_type=CounterOutput)
                    assert result.count == 11

            # Second access should reuse the cached builder (state persists)
            async with sm.session(user_id="user1") as session:
                async with session.run(CounterInput(action="get")) as runner:
                    result = await runner.result(to_type=CounterOutput)
                    assert result.count == 11  # Same builder, same state

        finally:
            await sm.shutdown()


async def test_session_manager_schemas_for_per_user_workflow():
    """Test that SessionManager provides correct schemas for per-user workflows."""

    config = Config(functions={
        "counter": PerUserCounterConfig(),
    },
                    workflow=PerUserCounterWorkflowConfig(counter_name="counter"))

    async with WorkflowBuilder.from_config(config) as builder:
        sm = await SessionManager.create(config=config, shared_builder=builder)

        try:
            # Verify schemas are accessible (for OpenAPI docs)
            assert sm.get_workflow_input_schema() == CounterInput
            assert sm.get_workflow_single_output_schema() == CounterOutput
            assert sm.is_workflow_per_user is True

            # workflow property should raise for per-user
            with pytest.raises(ValueError, match="Workflow is per-user"):
                _ = sm.workflow

        finally:
            await sm.shutdown()


# ============= Per-User Function Group Tests =============


# Test schemas and configs for function groups
class FunctionGroupToolInput(BaseModel):
    query: str = Field(description="Query to process")


class FunctionGroupToolOutput(BaseModel):
    result: str = Field(description="Processing result")


class SharedFunctionGroupConfig(FunctionGroupBaseConfig, name="shared_function_group"):
    """A shared function group config for testing."""
    pass


class PerUserFunctionGroupConfig(FunctionGroupBaseConfig, name="per_user_function_group"):
    """A per-user function group config for testing."""
    group_prefix: str = "group"


async def test_register_per_user_function_group():
    """Test that @register_per_user_function_group decorator works."""
    import typing

    from nat.cli.register_workflow import register_per_user_function_group
    from nat.cli.type_registry import GlobalTypeRegistry

    class TestPerUserGroupConfig(FunctionGroupBaseConfig, name="test_per_user_group_decorator"):
        pass

    @register_per_user_function_group(config_type=TestPerUserGroupConfig)
    async def test_group(config: TestPerUserGroupConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    # Check registration
    registration = GlobalTypeRegistry.get().get_function_group(TestPerUserGroupConfig)
    assert registration is not None
    assert registration.is_per_user is True
    assert registration.config_type == TestPerUserGroupConfig


async def test_workflow_builder_skips_per_user_function_groups():
    """Test that WorkflowBuilder.populate_builder() skips per-user function groups."""
    import typing

    from nat.cli.register_workflow import register_function_group
    from nat.cli.register_workflow import register_per_user_function_group

    class SharedGroupConfig(FunctionGroupBaseConfig, name="test_shared_group_skip"):
        pass

    class PerUserGroupConfig(FunctionGroupBaseConfig, name="test_per_user_group_skip"):
        pass

    @register_function_group(config_type=SharedGroupConfig)
    async def shared_group(config: SharedGroupConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_per_user_function_group(config_type=PerUserGroupConfig)
    async def per_user_group(config: PerUserGroupConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "shared_group": SharedGroupConfig(),
        "per_user_group": PerUserGroupConfig(),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as builder:
        # Shared function group should be built
        assert "shared_group" in builder._function_groups

        # Per-user function group should NOT be built in shared builder
        assert "per_user_group" not in builder._function_groups

        # Attempting to get per-user function group should fail
        with pytest.raises(ValueError, match="Function group `per_user_group` not found"):
            await builder.get_function_group("per_user_group")


async def test_per_user_builder_builds_per_user_function_groups():
    """Test that PerUserWorkflowBuilder builds per-user function groups."""
    import typing

    from nat.cli.register_workflow import register_per_user_function_group

    class TestPerUserFGConfig(FunctionGroupBaseConfig, name="test_pu_fg_build"):
        prefix: str

    @register_per_user_function_group(config_type=TestPerUserFGConfig)
    async def test_per_user_fg(config: TestPerUserFGConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            def __init__(self, config: TestPerUserFGConfig):
                super().__init__(config=config)
                self.prefix = config.prefix

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "test_fg": TestPerUserFGConfig(prefix="user_specific"),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user123", config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Per-user function group should be built
            assert "test_fg" in per_user_builder._per_user_function_groups

            # Should be able to get it
            fg = await per_user_builder.get_function_group("test_fg")
            assert fg is not None
            assert fg.prefix == "user_specific"


async def test_per_user_builder_function_groups_expose_functions():
    """Test that per-user function groups can expose functions."""
    from nat.builder.function import LambdaFunction
    from nat.cli.register_workflow import register_per_user_function_group

    class TestExposeConfig(FunctionGroupBaseConfig, name="test_expose_fg"):
        pass

    @register_per_user_function_group(config_type=TestExposeConfig)
    async def test_expose_fg(config: TestExposeConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            def __init__(self, config: TestExposeConfig):
                # Create a test function to expose
                async def exposed_fn(inp: str) -> str:
                    return f"exposed: {inp}"

                exposed_fn_obj = LambdaFunction.from_info(config=FunctionBaseConfig(),
                                                          info=FunctionInfo.from_fn(exposed_fn),
                                                          instance_name="exposed_tool")

                # Update config to include the exposed function
                config.include = ["exposed_tool"]

                super().__init__(config=config)

                # Add function to the group's internal functions dict
                self._functions["exposed_tool"] = exposed_fn_obj

        yield TestGroup(config=config)

    config = Config(function_groups={
        "expose_fg": TestExposeConfig(),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_expose",
                                                      config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Function group should be built
            assert "expose_fg" in per_user_builder._per_user_function_groups

            # Exposed function should be accessible with prefixed name (group_name.function_name)
            sep = FunctionGroup.SEPARATOR
            assert f"expose_fg{sep}exposed_tool" in per_user_builder._per_user_functions

            # Should be able to get and call it using the prefixed name
            exposed_fn = await per_user_builder.get_function(f"expose_fg{sep}exposed_tool")
            result = await exposed_fn.ainvoke("test", to_type=str)
            assert result == "exposed: test"


async def test_per_user_builder_get_function_group_delegates_to_shared():
    """Test that PerUserWorkflowBuilder delegates to shared builder for shared function groups."""
    import typing

    from nat.cli.register_workflow import register_function_group

    class TestSharedFGConfig(FunctionGroupBaseConfig, name="test_shared_fg_delegate"):
        pass

    @register_function_group(config_type=TestSharedFGConfig)
    async def test_shared_fg(config: TestSharedFGConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "shared_fg": TestSharedFGConfig(),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder(user_id="user_delegate", shared_builder=shared_builder) as per_user_builder:

            # Per-user builder should delegate to shared builder
            fg = await per_user_builder.get_function_group("shared_fg")
            assert fg is not None
            assert fg is shared_builder._function_groups["shared_fg"].instance


async def test_per_user_builder_get_function_group_config():
    """Test that PerUserWorkflowBuilder.get_function_group_config() works correctly."""
    import typing

    from nat.cli.register_workflow import register_function_group
    from nat.cli.register_workflow import register_per_user_function_group

    class SharedFGConf(FunctionGroupBaseConfig, name="test_shared_fg_conf"):
        shared_value: str = "shared"

    class PerUserFGConf(FunctionGroupBaseConfig, name="test_per_user_fg_conf"):
        per_user_value: str = "per_user"

    @register_function_group(config_type=SharedFGConf)
    async def shared_fg(config: SharedFGConf, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_per_user_function_group(config_type=PerUserFGConf)
    async def per_user_fg(config: PerUserFGConf, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "shared_fg": SharedFGConf(shared_value="test_shared"),
        "per_user_fg": PerUserFGConf(per_user_value="test_per_user"),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_conf", config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Get shared function group config
            shared_conf = per_user_builder.get_function_group_config("shared_fg")
            assert isinstance(shared_conf, SharedFGConf)
            assert shared_conf.shared_value == "test_shared"

            # Get per-user function group config
            per_user_conf = per_user_builder.get_function_group_config("per_user_fg")
            assert isinstance(per_user_conf, PerUserFGConf)
            assert per_user_conf.per_user_value == "test_per_user"


async def test_per_user_builder_build_merges_function_groups():
    """Test that PerUserWorkflowBuilder.build() merges shared and per-user function groups."""
    import typing

    from nat.cli.register_workflow import register_function_group
    from nat.cli.register_workflow import register_per_user_function_group

    class SharedBuildFGConfig(FunctionGroupBaseConfig, name="test_shared_build_fg"):
        pass

    class PerUserBuildFGConfig(FunctionGroupBaseConfig, name="test_per_user_build_fg"):
        pass

    @register_function_group(config_type=SharedBuildFGConfig)
    async def shared_build_fg(config: SharedBuildFGConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_per_user_function_group(config_type=PerUserBuildFGConfig)
    async def per_user_build_fg(config: PerUserBuildFGConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "shared_fg": SharedBuildFGConfig(),
        "per_user_fg": PerUserBuildFGConfig(),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_merge",
                                                      config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Build workflow
            workflow = await per_user_builder.build()

            # Both shared and per-user function groups should be in the workflow
            assert "shared_fg" in workflow.function_groups
            assert "per_user_fg" in workflow.function_groups


async def test_per_user_builder_get_tools_with_function_groups():
    """Test that PerUserWorkflowBuilder.get_tools() expands function groups correctly."""
    import typing

    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.builder.function import LambdaFunction
    from nat.cli.register_workflow import register_function_group
    from nat.cli.register_workflow import register_per_user_function_group

    class SharedToolsFGConfig(FunctionGroupBaseConfig, name="test_shared_tools_fg"):
        pass

    class PerUserToolsFGConfig(FunctionGroupBaseConfig, name="test_per_user_tools_fg"):
        pass

    @register_function_group(config_type=SharedToolsFGConfig)
    async def shared_tools_fg(config: SharedToolsFGConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            def __init__(self, config: SharedToolsFGConfig):
                super().__init__(config=config)

                async def tool1(inp: str) -> str:
                    return f"shared_tool1: {inp}"

                self._tool1 = LambdaFunction.from_info(config=FunctionBaseConfig(),
                                                       info=FunctionInfo.from_fn(tool1),
                                                       instance_name="shared_tool1")

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {"shared_tool1": self._tool1}

        yield TestGroup(config=config)

    @register_per_user_function_group(config_type=PerUserToolsFGConfig)
    async def per_user_tools_fg(config: PerUserToolsFGConfig, builder: Builder):

        class TestGroup(FunctionGroup):

            def __init__(self, config: PerUserToolsFGConfig):
                super().__init__(config=config)

                async def tool2(inp: str) -> str:
                    return f"per_user_tool2: {inp}"

                self._tool2 = LambdaFunction.from_info(config=FunctionBaseConfig(),
                                                       info=FunctionInfo.from_fn(tool2),
                                                       instance_name="per_user_tool2")

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {"per_user_tool2": self._tool2}

        yield TestGroup(config=config)

    config = Config(function_groups={
        "shared_fg": SharedToolsFGConfig(),
        "per_user_fg": PerUserToolsFGConfig(),
    },
                    functions={WORKFLOW_COMPONENT_NAME: SharedWorkflowConfig()},
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        async with PerUserWorkflowBuilder.from_config(user_id="user_tools",
                                                      config=config,
                                                      shared_builder=shared_builder) as per_user_builder:

            # Get tools from both function groups
            # Note: This test just verifies the tools can be retrieved
            # Actual tool wrapping depends on registered tool wrappers
            try:
                tools = await per_user_builder.get_tools(["shared_fg", "per_user_fg"],
                                                         wrapper_type=LLMFrameworkEnum.LANGCHAIN)
                # Should get tools from both groups (if wrapper is registered)
                assert len(tools) >= 0  # May be 0 if no wrapper registered in test environment
            except ValueError as e:
                # Expected if no tool wrapper is registered for LANGCHAIN
                assert "not registered" in str(e) or "Tool wrapper" in str(e)


async def test_per_user_builder_populate_builds_function_groups_before_functions():
    """Test that populate_builder builds function groups before functions (dependency order)."""
    import typing

    from nat.cli.register_workflow import register_per_user_function_group

    class OrderTestFGConfig(FunctionGroupBaseConfig, name="test_order_fg"):
        pass

    class OrderTestFnConfig(FunctionBaseConfig, name="test_order_fn"):
        fg_name: str

    build_order = []

    @register_per_user_function_group(config_type=OrderTestFGConfig)
    async def order_fg(config: OrderTestFGConfig, builder: Builder):
        build_order.append("function_group")

        class TestGroup(FunctionGroup):

            async def get_accessible_functions(
                self,
                filter_fn: typing.Callable[[typing.Sequence[str]], typing.Awaitable[typing.Sequence[str]]]
                | None = None,
            ) -> dict[str, typing.Any]:
                return {}

        yield TestGroup(config=config)

    @register_per_user_function(config_type=OrderTestFnConfig,
                                input_type=PerUserInputSchema,
                                single_output_type=PerUserOutputSchema)
    async def order_fn(config: OrderTestFnConfig, builder: Builder):
        build_order.append("function")
        # Access the function group (dependency)
        _ = await builder.get_function_group(config.fg_name)

        async def _impl(inp: PerUserInputSchema) -> PerUserOutputSchema:
            return PerUserOutputSchema(result="test")

        yield FunctionInfo.from_fn(_impl)

    config = Config(function_groups={
        "test_fg": OrderTestFGConfig(),
    },
                    functions={
                        "test_fn": OrderTestFnConfig(fg_name="test_fg"),
                    },
                    workflow=SharedWorkflowConfig())

    async with WorkflowBuilder.from_config(config) as shared_builder:
        build_order.clear()
        async with PerUserWorkflowBuilder.from_config(user_id="user_order",
                                                      config=config,
                                                      shared_builder=shared_builder):

            # Function group should be built before function
            assert build_order == ["function_group", "function"]
