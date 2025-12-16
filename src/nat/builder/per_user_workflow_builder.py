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
import logging
import typing
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import cast

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.builder import UserManagerHolder
from nat.builder.child_builder import ChildBuilder
from nat.builder.component_utils import WORKFLOW_COMPONENT_NAME
from nat.builder.component_utils import build_dependency_sequence
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import ConfiguredFunction
from nat.builder.workflow_builder import ConfiguredFunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.builder.workflow_builder import _build_function_group_impl
from nat.builder.workflow_builder import _build_function_impl
from nat.builder.workflow_builder import _log_build_failure
from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import TypeRegistry
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.component import ComponentGroup
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.component_ref import TrainerAdapterRef
from nat.data_models.component_ref import TrainerRef
from nat.data_models.component_ref import TrajectoryBuilderRef
from nat.data_models.component_ref import TTCStrategyRef
from nat.data_models.config import Config
from nat.data_models.embedder import EmbedderBaseConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.function_dependencies import FunctionDependencies
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.middleware import MiddlewareBaseConfig
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.data_models.retriever import RetrieverBaseConfig
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.finetuning.interfaces.finetuning_runner import Trainer
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder
from nat.memory.interfaces import MemoryEditor
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import Middleware
from nat.object_store.interfaces import ObjectStore
from nat.retriever.interface import Retriever
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class PerUserWorkflowBuilder(Builder, AbstractAsyncContextManager):
    """
    Builder for per-user components that are lazily instantiated.

    This builder is created per-user and only builds functions/function_groups
    that are marked as per-user. It delegates to a shared WorkflowBuilder for
    all shared components (LLMs, embedders, memory, etc.).

    Lifecycle:
    - Created when a user first makes a request
    - Kept alive while the user is active
    - Cleaned up after user inactivity timeout
    """

    def __init__(self, user_id: str, shared_builder: WorkflowBuilder, registry: TypeRegistry | None = None):

        self._user_id = user_id
        self._shared_builder = shared_builder
        self._workflow: ConfiguredFunction | None = None

        if registry is None:
            registry = GlobalTypeRegistry.get()
        self._registry = registry

        self._per_user_functions: dict[str, ConfiguredFunction] = {}
        self._per_user_function_groups: dict[str, ConfiguredFunctionGroup] = {}

        self._exit_stack: AsyncExitStack | None = None

        self.per_user_function_dependencies: dict[str, FunctionDependencies] = {}
        self.per_user_function_group_dependencies: dict[str, FunctionDependencies] = {}

        # Copy the completed and remaining components from the shared builder
        self.completed_components: list[tuple[str, str]] = shared_builder.completed_components.copy()
        self.remaining_components: list[tuple[str, str]] = shared_builder.remaining_components.copy()

    async def __aenter__(self):

        self._exit_stack = AsyncExitStack()
        return self

    async def __aexit__(self, *exc_details):

        assert self._exit_stack is not None, "Exit stack not initialized"
        await self._exit_stack.__aexit__(*exc_details)

    def _get_exit_stack(self) -> AsyncExitStack:
        if self._exit_stack is None:
            raise ValueError(
                "Exit stack not initialized. Did you forget to call `async with PerUserWorkflowBuilder() as builder`?")
        return self._exit_stack

    @property
    def user_id(self) -> str:
        return self._user_id

    async def _resolve_middleware_instances_from_shared_builder(self,
                                                                middleware_names: Sequence[str],
                                                                component_type: str = "function"
                                                                ) -> list[FunctionMiddleware]:
        """
        Resolve middleware names to FunctionMiddleware instances from the shared builder.
        """
        middleware_instances: list[FunctionMiddleware] = []
        for middleware_name in middleware_names:
            middleware_obj = await self._shared_builder.get_middleware(middleware_name)
            if not isinstance(middleware_obj, FunctionMiddleware):
                raise TypeError(f"Middleware `{middleware_name}` is not a FunctionMiddleware and cannot be used "
                                f"with {component_type}s. "
                                f"Only FunctionMiddleware types support function-specific wrapping.")
            middleware_instances.append(middleware_obj)
        return middleware_instances

    async def _build_per_user_function(self, name: str, config: FunctionBaseConfig) -> ConfiguredFunction:
        registration = self._registry.get_function(type(config))

        if not registration.is_per_user:
            raise ValueError(f"Function `{name}` is not a per-user function")

        inner_builder = ChildBuilder(self)

        llms = {k: v.instance for k, v in self._shared_builder._llms.items()}
        middleware_instances = await self._resolve_middleware_instances_from_shared_builder(
            config.middleware, "function")
        return await _build_function_impl(name=name,
                                          config=config,
                                          registry=self._registry,
                                          exit_stack=self._get_exit_stack(),
                                          inner_builder=inner_builder,
                                          llms=llms,
                                          dependencies=self.per_user_function_dependencies,
                                          middleware_instances=middleware_instances)

    async def _build_per_user_function_group(self, name: str,
                                             config: FunctionGroupBaseConfig) -> ConfiguredFunctionGroup:
        registration = self._registry.get_function_group(type(config))

        if not registration.is_per_user:
            raise ValueError(f"Function group `{name}` is not a per-user function group")

        inner_builder = ChildBuilder(self)

        llms = {k: v.instance for k, v in self._shared_builder._llms.items()}
        middleware_instances = await self._resolve_middleware_instances_from_shared_builder(
            config.middleware, "function group")

        return await _build_function_group_impl(name=name,
                                                config=config,
                                                registry=self._registry,
                                                exit_stack=self._get_exit_stack(),
                                                inner_builder=inner_builder,
                                                llms=llms,
                                                dependencies=self.per_user_function_group_dependencies,
                                                middleware_instances=middleware_instances)

    @override
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:
        if isinstance(name, FunctionRef):
            name = str(name)

        if name in self._per_user_functions:
            raise ValueError(f"Function `{name}` already exists in the list of per-user functions")

        registration = self._registry.get_function(type(config))
        if registration.is_per_user:
            build_result = await self._build_per_user_function(name, config)
            self._per_user_functions[name] = build_result

            return build_result.instance

        return await self._shared_builder.get_function(name)

    @override
    async def get_function(self, name: str | FunctionRef) -> Function:
        if isinstance(name, FunctionRef):
            name = str(name)

        # Check per-user cache first
        if name in self._per_user_functions:
            return self._per_user_functions[name].instance

        # Delegate to shared builder
        return await self._shared_builder.get_function(name)

    @override
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        if isinstance(name, FunctionRef):
            name = str(name)

        if name in self._per_user_functions:
            return self._per_user_functions[name].config

        return self._shared_builder.get_function_config(name)

    @override
    async def add_function_group(self, name: str | FunctionGroupRef, config: FunctionGroupBaseConfig) -> FunctionGroup:
        if isinstance(name, FunctionGroupRef):
            name = str(name)

        if (name in self._per_user_function_groups or name in self._per_user_functions):
            raise ValueError(f"Function group `{name}` already exists in the list of function groups or functions")

        registration = self._registry.get_function_group(type(config))
        if registration.is_per_user:
            # Build the per-user function group
            build_result = await self._build_per_user_function_group(name=name, config=config)

            self._per_user_function_groups[name] = build_result

            # If the function group exposes functions, add them to the per-user function registry
            included_functions = await build_result.instance.get_included_functions()
            for k in included_functions:
                if k in self._per_user_functions:
                    raise ValueError(f"Exposed function `{k}` from group `{name}` conflicts with an existing function")
            self._per_user_functions.update({
                k: ConfiguredFunction(config=v.config, instance=v)
                for k, v in included_functions.items()
            })

            return build_result.instance
        else:
            # Shared function group - delegate to shared builder
            return await self._shared_builder.get_function_group(name)

    @override
    async def get_function_group(self, name: str | FunctionGroupRef) -> FunctionGroup:
        if isinstance(name, FunctionGroupRef):
            name = str(name)

        # Check per-user function groups first
        if name in self._per_user_function_groups:
            return self._per_user_function_groups[name].instance

        # Fall back to shared builder for shared function groups
        return await self._shared_builder.get_function_group(name)

    @override
    def get_function_group_config(self, name: str | FunctionGroupRef) -> FunctionGroupBaseConfig:
        if isinstance(name, FunctionGroupRef):
            name = str(name)

        # Check per-user function groups first
        if name in self._per_user_function_groups:
            return self._per_user_function_groups[name].config

        # Fall back to shared builder
        return self._shared_builder.get_function_group_config(name)

    @override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        if self._workflow is not None:
            logger.warning("Overwriting existing workflow")

        build_result = await self._build_per_user_function(name=WORKFLOW_COMPONENT_NAME, config=config)

        self._workflow = build_result

        return build_result.instance

    @override
    def get_workflow(self) -> Function:
        # If we have a per-user workflow, return it
        if self._workflow is not None:
            return self._workflow.instance

        # Otherwise, delegate to shared builder
        return self._shared_builder.get_workflow()

    @override
    def get_workflow_config(self) -> FunctionBaseConfig:
        # If we have a per-user workflow config, return it
        if self._workflow is not None:
            return self._workflow.config

        # Otherwise, delegate to shared builder
        return self._shared_builder.get_workflow_config()

    @override
    def get_function_dependencies(self, fn_name: str | FunctionRef) -> FunctionDependencies:
        if isinstance(fn_name, FunctionRef):
            fn_name = str(fn_name)

        if fn_name in self.per_user_function_dependencies:
            return self.per_user_function_dependencies[fn_name]
        return self._shared_builder.get_function_dependencies(fn_name)

    @override
    def get_function_group_dependencies(self, fn_name: str | FunctionGroupRef) -> FunctionDependencies:
        if isinstance(fn_name, FunctionGroupRef):
            fn_name = str(fn_name)

        # Check per-user dependencies first
        if fn_name in self.per_user_function_group_dependencies:
            return self.per_user_function_group_dependencies[fn_name]

        # Fall back to shared builder
        return self._shared_builder.get_function_group_dependencies(fn_name)

    @override
    async def get_tools(self,
                        tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                        wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        unique = set(tool_names)
        if len(unique) != len(tool_names):
            raise ValueError("Tool names must be unique")

        async def _get_tools(n: str | FunctionRef | FunctionGroupRef):
            tools = []
            is_function_group_ref = isinstance(n, FunctionGroupRef)
            if isinstance(n, FunctionRef) or is_function_group_ref:
                n = str(n)

            # Check per-user function groups first
            if n not in self._per_user_function_groups:
                # Check shared function groups
                if n not in self._shared_builder._function_groups:
                    # The passed tool name is probably a function, but first check if it's a function group
                    if is_function_group_ref:
                        raise ValueError(f"Function group `{n}` not found in the list of function groups")
                    tools.append(await self.get_tool(n, wrapper_type))
                else:
                    # It's a shared function group
                    tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
                    current_function_group = self._shared_builder._function_groups[n]
                    for fn_name, fn_instance in \
                                            (await current_function_group.instance.get_accessible_functions()).items():
                        try:
                            tools.append(tool_wrapper_reg.build_fn(fn_name, fn_instance, self))
                        except Exception:
                            logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
                            raise
            else:
                # It's a per-user function group
                tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
                current_function_group = self._per_user_function_groups[n]
                for fn_name, fn_instance in (await current_function_group.instance.get_accessible_functions()).items():
                    try:
                        tools.append(tool_wrapper_reg.build_fn(fn_name, fn_instance, self))
                    except Exception:
                        logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
                        raise
            return tools

        tool_lists = await asyncio.gather(*[_get_tools(n) for n in tool_names])
        # Flatten the list of lists into a single list
        return [tool for sublist in tool_lists for tool in sublist]

    @override
    async def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        if isinstance(fn_name, FunctionRef):
            fn_name = str(fn_name)
        if fn_name in self._per_user_functions:
            fn = self._per_user_functions[fn_name]
            try:
                tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
                return tool_wrapper_reg.build_fn(fn_name, fn.instance, self)
            except Exception as e:
                logger.error("Error fetching tool `%s`: %s", fn_name, e)
                raise
        return await self._shared_builder.get_tool(fn_name, wrapper_type)

    @override
    async def add_llm(self, name: str, config: LLMBaseConfig) -> None:
        return await self._shared_builder.add_llm(name, config)

    @override
    async def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        return await self._shared_builder.get_llm(llm_name, wrapper_type)

    @override
    def get_llm_config(self, llm_name: str) -> LLMBaseConfig:
        return self._shared_builder.get_llm_config(llm_name)

    @experimental(feature_name="Authentication")
    @override
    async def add_auth_provider(self, name: str, config: AuthProviderBaseConfig) -> AuthProviderBase:
        return await self._shared_builder.add_auth_provider(name, config)

    @override
    async def get_auth_provider(self, auth_provider_name: str) -> AuthProviderBase:
        return await self._shared_builder.get_auth_provider(auth_provider_name)

    @override
    async def add_embedder(self, name: str, config: EmbedderBaseConfig) -> None:
        return await self._shared_builder.add_embedder(name, config)

    @override
    async def get_embedder(self, embedder_name: str, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        return await self._shared_builder.get_embedder(embedder_name, wrapper_type)

    @override
    def get_embedder_config(self, embedder_name: str) -> EmbedderBaseConfig:
        return self._shared_builder.get_embedder_config(embedder_name)

    @override
    async def add_memory_client(self, name: str, config: MemoryBaseConfig) -> MemoryEditor:
        return await self._shared_builder.add_memory_client(name, config)

    @override
    async def get_memory_client(self, memory_name: str) -> MemoryEditor:
        return await self._shared_builder.get_memory_client(memory_name)

    @override
    def get_memory_client_config(self, memory_name: str) -> MemoryBaseConfig:
        return self._shared_builder.get_memory_client_config(memory_name)

    @override
    async def add_object_store(self, name: str, config: ObjectStoreBaseConfig) -> ObjectStore:
        return await self._shared_builder.add_object_store(name, config)

    @override
    async def get_object_store_client(self, object_store_name: str) -> ObjectStore:
        return await self._shared_builder.get_object_store_client(object_store_name)

    @override
    def get_object_store_config(self, object_store_name: str) -> ObjectStoreBaseConfig:
        return self._shared_builder.get_object_store_config(object_store_name)

    @override
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig) -> None:
        return await self._shared_builder.add_retriever(name, config)

    @override
    async def get_retriever(self,
                            retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str | None = None) -> Retriever:
        return await self._shared_builder.get_retriever(retriever_name, wrapper_type)

    @override
    async def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:
        return await self._shared_builder.get_retriever_config(retriever_name)

    @experimental(feature_name="TTC")
    @override
    async def add_ttc_strategy(self, name: str | TTCStrategyRef, config: TTCStrategyBaseConfig) -> None:
        return await self._shared_builder.add_ttc_strategy(name, config)

    @override
    async def get_ttc_strategy(self,
                               strategy_name: str | TTCStrategyRef,
                               pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum) -> StrategyBase:
        return await self._shared_builder.get_ttc_strategy(strategy_name, pipeline_type, stage_type)

    @override
    async def get_ttc_strategy_config(self,
                                      strategy_name: str | TTCStrategyRef,
                                      pipeline_type: PipelineTypeEnum,
                                      stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        return await self._shared_builder.get_ttc_strategy_config(strategy_name, pipeline_type, stage_type)

    @override
    def get_user_manager(self) -> UserManagerHolder:
        return self._shared_builder.get_user_manager()

    @override
    async def add_middleware(self, name: str | MiddlewareRef, config: MiddlewareBaseConfig) -> Middleware:
        return await self._shared_builder.add_middleware(name, config)

    @override
    async def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        return await self._shared_builder.get_middleware(middleware_name)

    @override
    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        return self._shared_builder.get_middleware_config(middleware_name)

    @experimental(feature_name="Finetuning")
    @override
    async def add_trainer(self, name: str | TrainerRef, config: TrainerConfig) -> Trainer:
        return await self._shared_builder.add_trainer(name, config)

    @experimental(feature_name="Finetuning")
    @override
    async def add_trainer_adapter(self, name: str | TrainerAdapterRef, config: TrainerAdapterConfig) -> TrainerAdapter:
        return await self._shared_builder.add_trainer_adapter(name, config)

    @experimental(feature_name="Finetuning")
    @override
    async def add_trajectory_builder(self, name: str | TrajectoryBuilderRef,
                                     config: TrajectoryBuilderConfig) -> TrajectoryBuilder:
        return await self._shared_builder.add_trajectory_builder(name, config)

    @override
    async def get_trainer(self,
                          trainer_name: str | TrainerRef,
                          trajectory_builder: TrajectoryBuilder,
                          trainer_adapter: TrainerAdapter) -> Trainer:
        return await self._shared_builder.get_trainer(trainer_name, trajectory_builder, trainer_adapter)

    @override
    async def get_trainer_adapter(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapter:
        return await self._shared_builder.get_trainer_adapter(trainer_adapter_name)

    @override
    async def get_trajectory_builder(self, trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilder:
        return await self._shared_builder.get_trajectory_builder(trajectory_builder_name)

    @override
    async def get_trainer_config(self, trainer_name: str | TrainerRef) -> TrainerConfig:
        return await self._shared_builder.get_trainer_config(trainer_name)

    @override
    async def get_trainer_adapter_config(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapterConfig:
        return await self._shared_builder.get_trainer_adapter_config(trainer_adapter_name)

    @override
    async def get_trajectory_builder_config(
            self, trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilderConfig:
        return await self._shared_builder.get_trajectory_builder_config(trajectory_builder_name)

    async def populate_builder(self, config: Config, skip_workflow: bool = False):
        """
        Populate the per-user builder with per-user components from config.

        Only builds components that are marked as per-user.
        Builds in dependency order to handle per-user functions depending on other per-user functions.

        Args:
            config: The full configuration object
            skip_workflow: If True, skips the workflow instantiation step. Defaults to False.
        Raises:
            ValueError: If a per-user component has invalid dependencies
        """
        # Generate build sequence using the same dependency resolution as shared builder
        build_sequence = build_dependency_sequence(config)

        if not skip_workflow:
            if (WORKFLOW_COMPONENT_NAME, "workflow") not in self.remaining_components:
                self.remaining_components.append((WORKFLOW_COMPONENT_NAME, "workflow"))

        # Filter to only per-user functions and function groups and build them in dependency order
        for component_instance in build_sequence:
            try:
                if component_instance.component_group == ComponentGroup.FUNCTION_GROUPS:
                    config_obj = cast(FunctionGroupBaseConfig, component_instance.config)
                    registration = self._registry.get_function_group(type(config_obj))
                    if registration.is_per_user:
                        # Build the per-user function group
                        logger.debug(
                            f"Building per-user function group '{component_instance.name}' for user {self._user_id}")
                        await self.add_function_group(component_instance.name, config_obj)
                        self.remaining_components.remove(
                            (str(component_instance.name), component_instance.component_group.value))
                        self.completed_components.append(
                            (str(component_instance.name), component_instance.component_group.value))
                    else:
                        continue

                elif component_instance.component_group == ComponentGroup.FUNCTIONS:
                    config_obj = cast(FunctionBaseConfig, component_instance.config)
                    registration = self._registry.get_function(type(config_obj))
                    if registration.is_per_user:
                        if not component_instance.is_root:
                            logger.debug(
                                f"Building per-user function '{component_instance.name}' for user {self._user_id}")
                            await self.add_function(component_instance.name, config_obj)
                            self.remaining_components.remove(
                                (str(component_instance.name), component_instance.component_group.value))
                            self.completed_components.append(
                                (str(component_instance.name), component_instance.component_group.value))
                    else:
                        continue

            except Exception as e:
                _log_build_failure(str(component_instance.name),
                                   component_instance.component_group.value,
                                   self.completed_components,
                                   self.remaining_components,
                                   e)
                raise

        if not skip_workflow:
            try:
                registration = self._registry.get_function(type(config.workflow))
                if registration.is_per_user:
                    self.remaining_components.remove((WORKFLOW_COMPONENT_NAME, "workflow"))
                    await self.set_workflow(config.workflow)
                    self.completed_components.append((WORKFLOW_COMPONENT_NAME, "workflow"))
            except Exception as e:
                _log_build_failure(WORKFLOW_COMPONENT_NAME,
                                   "workflow",
                                   self.completed_components,
                                   self.remaining_components,
                                   e)
                raise

    async def build(self, entry_function: str | None = None) -> Workflow:
        """
        Creates a workflow instance for this specific user.

        Combines per-user functions with shared components from the shared builder.

        Parameters
        ----------
        entry_function : str | None, optional
            The function name to use as the entry point. If None, uses the workflow.
            By default None

        Returns
        -------
        Workflow
            A per-user workflow instance

        Raises
        ------
        ValueError
            If no workflow is set (neither per-user nor shared)
        """
        # Determine entry function
        if entry_function is None:
            # Use workflow (could be per-user or shared)
            entry_fn_obj = self.get_workflow()
        else:
            # Use specified function (could be per-user or shared)
            entry_fn_obj = await self.get_function(entry_function)

        # Collect function names that are included by function groups (shared + per-user)
        # These will be skipped when populating function_configs and all_functions
        included_functions: set[str] = set()
        for configured_fg in self._shared_builder._function_groups.values():
            included_functions.update((await configured_fg.instance.get_included_functions()).keys())
        for configured_fg in self._per_user_function_groups.values():
            included_functions.update((await configured_fg.instance.get_included_functions()).keys())

        # Collect all functions (per-user + shared), excluding those already in function groups
        all_functions = {}

        # Add shared functions (skip those included by function groups)
        for name, configured_fn in self._shared_builder._functions.items():
            if name not in included_functions:
                all_functions[name] = configured_fn.instance

        # Override with per-user functions (skip those included by function groups)
        for name, configured_fn in self._per_user_functions.items():
            if name not in included_functions:
                all_functions[name] = configured_fn.instance

        # Collect all function groups (shared + per-user)
        all_function_groups = {}
        # Add shared function groups
        for name, configured_fg in self._shared_builder._function_groups.items():
            all_function_groups[name] = configured_fg.instance
        # Override with per-user function groups
        for name, configured_fg in self._per_user_function_groups.items():
            all_function_groups[name] = configured_fg.instance

        # Build function configs (per-user + shared), excluding those already in function groups
        function_configs = {}
        for name, configured_fn in self._shared_builder._functions.items():
            if name not in included_functions:
                function_configs[name] = configured_fn.config
        for name, configured_fn in self._per_user_functions.items():
            if name not in included_functions:
                function_configs[name] = configured_fn.config

        # Build function group configs (shared + per-user)
        function_group_configs = {}
        for name, configured_fg in self._shared_builder._function_groups.items():
            function_group_configs[name] = configured_fg.config
        for name, configured_fg in self._per_user_function_groups.items():
            function_group_configs[name] = configured_fg.config

        # Determine workflow config
        if self._workflow is not None:
            workflow_config = self._workflow.config
        else:
            workflow_config = self._shared_builder.get_workflow_config()

        # Build the Config object
        per_user_config = Config(general=self._shared_builder.general_config,
                                 functions=function_configs,
                                 function_groups=function_group_configs,
                                 workflow=workflow_config,
                                 llms={
                                     k: v.config
                                     for k, v in self._shared_builder._llms.items()
                                 },
                                 embedders={
                                     k: v.config
                                     for k, v in self._shared_builder._embedders.items()
                                 },
                                 memory={
                                     k: v.config
                                     for k, v in self._shared_builder._memory_clients.items()
                                 },
                                 object_stores={
                                     k: v.config
                                     for k, v in self._shared_builder._object_stores.items()
                                 },
                                 retrievers={
                                     k: v.config
                                     for k, v in self._shared_builder._retrievers.items()
                                 },
                                 ttc_strategies={
                                     k: v.config
                                     for k, v in self._shared_builder._ttc_strategies.items()
                                 })

        # Create the Workflow instance
        workflow = Workflow.from_entry_fn(config=per_user_config,
                                          entry_fn=entry_fn_obj,
                                          functions=all_functions,
                                          function_groups=all_function_groups,
                                          llms={
                                              k: v.instance
                                              for k, v in self._shared_builder._llms.items()
                                          },
                                          embeddings={
                                              k: v.instance
                                              for k, v in self._shared_builder._embedders.items()
                                          },
                                          memory={
                                              k: v.instance
                                              for k, v in self._shared_builder._memory_clients.items()
                                          },
                                          object_stores={
                                              k: v.instance
                                              for k, v in self._shared_builder._object_stores.items()
                                          },
                                          telemetry_exporters={
                                              k: v.instance
                                              for k, v in self._shared_builder._telemetry_exporters.items()
                                          },
                                          retrievers={
                                              k: v.instance
                                              for k, v in self._shared_builder._retrievers.items()
                                          },
                                          ttc_strategies={
                                              k: v.instance
                                              for k, v in self._shared_builder._ttc_strategies.items()
                                          },
                                          context_state=self._shared_builder._context_state)

        return workflow

    @classmethod
    @asynccontextmanager
    async def from_config(cls, user_id: str, config: Config, shared_builder: WorkflowBuilder):
        """
        Create and populate a PerUserWorkflowBuilder from config.

        This is the primary entry point for creating per-user builders.

        Args:
            user_id: Unique identifier for the user
            config: Full configuration object
            shared_builder: The shared WorkflowBuilder instance

        Yields:
            PerUserWorkflowBuilder: Populated per-user builder instance
        """
        async with cls(user_id=user_id, shared_builder=shared_builder) as builder:
            await builder.populate_builder(config)
            yield builder
