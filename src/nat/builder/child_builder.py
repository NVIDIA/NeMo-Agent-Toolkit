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

import typing
from collections.abc import Sequence

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.builder import UserManagerHolder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.component_ref import TrainerAdapterRef
from nat.data_models.component_ref import TrainerRef
from nat.data_models.component_ref import TrajectoryBuilderRef
from nat.data_models.component_ref import TTCStrategyRef
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
from nat.middleware.middleware import Middleware
from nat.object_store.interfaces import ObjectStore
from nat.retriever.interface import Retriever
from nat.utils.type_utils import override


class ChildBuilder(Builder):

    def __init__(self, workflow_builder: Builder) -> None:

        self._workflow_builder = workflow_builder

        self._dependencies = FunctionDependencies()

    @property
    def dependencies(self) -> FunctionDependencies:
        return self._dependencies

    @override
    async def add_function(self, name: str, config: FunctionBaseConfig) -> Function:
        return await self._workflow_builder.add_function(name, config)

    @override
    async def add_function_group(self, name: str, config: FunctionGroupBaseConfig) -> FunctionGroup:
        return await self._workflow_builder.add_function_group(name, config)

    @override
    async def get_function(self, name: str) -> Function:
        # If a function tries to get another function, we assume it uses it
        fn = await self._workflow_builder.get_function(name)

        self._dependencies.add_function(name)

        return fn

    @override
    async def get_function_group(self, name: str) -> FunctionGroup:
        # If a function tries to get a function group, we assume it uses it
        function_group = await self._workflow_builder.get_function_group(name)

        self._dependencies.add_function_group(name)

        return function_group

    @override
    def get_function_config(self, name: str) -> FunctionBaseConfig:
        return self._workflow_builder.get_function_config(name)

    @override
    def get_function_group_config(self, name: str) -> FunctionGroupBaseConfig:
        return self._workflow_builder.get_function_group_config(name)

    @override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        return await self._workflow_builder.set_workflow(config)

    @override
    def get_workflow(self) -> Function:
        return self._workflow_builder.get_workflow()

    @override
    def get_workflow_config(self) -> FunctionBaseConfig:
        return self._workflow_builder.get_workflow_config()

    @override
    async def get_tools(self,
                        tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                        wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        # Import here to avoid cyclic import
        from nat.builder.per_user_workflow_builder import PerUserWorkflowBuilder
        from nat.builder.workflow_builder import WorkflowBuilder

        tools = await self._workflow_builder.get_tools(tool_names, wrapper_type)
        for tool_name in tool_names:
            if isinstance(self._workflow_builder, WorkflowBuilder):
                function_groups = self._workflow_builder._function_groups
            elif isinstance(self._workflow_builder, PerUserWorkflowBuilder):
                # Per-user components can have dependencies on both shared and per-user function groups
                function_groups = {
                    **self._workflow_builder._shared_builder._function_groups,
                    **self._workflow_builder._per_user_function_groups
                }
            else:
                raise TypeError(f"Invalid workflow builder type: {type(self._workflow_builder)}")
            if tool_name in function_groups:
                self._dependencies.add_function_group(tool_name)
            else:
                self._dependencies.add_function(tool_name)
        return tools

    @override
    async def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str):
        # If a function tries to get another function as a tool, we assume it uses it
        fn = await self._workflow_builder.get_tool(fn_name, wrapper_type)

        self._dependencies.add_function(fn_name)

        return fn

    @override
    async def add_llm(self, name: str, config: LLMBaseConfig) -> None:
        return await self._workflow_builder.add_llm(name, config)

    @experimental(feature_name="Authentication")
    @override
    async def add_auth_provider(self, name: str, config: AuthProviderBaseConfig) -> AuthProviderBase:
        return await self._workflow_builder.add_auth_provider(name, config)

    @override
    async def get_auth_provider(self, auth_provider_name: str):
        return await self._workflow_builder.get_auth_provider(auth_provider_name)

    @override
    async def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        llm = await self._workflow_builder.get_llm(llm_name, wrapper_type)

        self._dependencies.add_llm(llm_name)

        return llm

    @override
    def get_llm_config(self, llm_name: str) -> LLMBaseConfig:
        return self._workflow_builder.get_llm_config(llm_name)

    @override
    async def add_embedder(self, name: str, config: EmbedderBaseConfig) -> None:
        await self._workflow_builder.add_embedder(name, config)

    @override
    async def get_embedder(self, embedder_name: str, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        embedder = await self._workflow_builder.get_embedder(embedder_name, wrapper_type)

        self._dependencies.add_embedder(embedder_name)

        return embedder

    @override
    def get_embedder_config(self, embedder_name: str) -> EmbedderBaseConfig:
        return self._workflow_builder.get_embedder_config(embedder_name)

    @override
    async def add_memory_client(self, name: str, config: MemoryBaseConfig) -> MemoryEditor:
        return await self._workflow_builder.add_memory_client(name, config)

    @override
    async def get_memory_client(self, memory_name: str) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        memory_client = await self._workflow_builder.get_memory_client(memory_name)

        self._dependencies.add_memory_client(memory_name)

        return memory_client

    @override
    def get_memory_client_config(self, memory_name: str) -> MemoryBaseConfig:
        return self._workflow_builder.get_memory_client_config(memory_name=memory_name)

    @override
    async def add_object_store(self, name: str, config: ObjectStoreBaseConfig):
        return await self._workflow_builder.add_object_store(name, config)

    @override
    async def get_object_store_client(self, object_store_name: str) -> ObjectStore:
        """
        Return the instantiated object store client for the given name.
        """
        object_store_client = await self._workflow_builder.get_object_store_client(object_store_name)

        self._dependencies.add_object_store(object_store_name)

        return object_store_client

    @override
    def get_object_store_config(self, object_store_name: str) -> ObjectStoreBaseConfig:
        return self._workflow_builder.get_object_store_config(object_store_name)

    @override
    @experimental(feature_name="Finetuning")
    async def add_trainer(self, name: str | TrainerRef, config: TrainerConfig) -> Trainer:
        return await self._workflow_builder.add_trainer(name, config)

    @override
    @experimental(feature_name="Finetuning")
    async def add_trainer_adapter(self, name: str | TrainerAdapterRef, config: TrainerAdapterConfig) -> TrainerAdapter:
        return await self._workflow_builder.add_trainer_adapter(name, config)

    @override
    @experimental(feature_name="Finetuning")
    async def add_trajectory_builder(self, name: str | TrajectoryBuilderRef,
                                     config: TrajectoryBuilderConfig) -> TrajectoryBuilder:
        return await self._workflow_builder.add_trajectory_builder(name, config)

    @override
    async def get_trainer(self,
                          trainer_name: str | TrainerRef,
                          trajectory_builder: TrajectoryBuilder,
                          trainer_adapter: TrainerAdapter) -> Trainer:
        return await self._workflow_builder.get_trainer(trainer_name, trajectory_builder, trainer_adapter)

    @override
    async def get_trainer_config(self, trainer_name: str | TrainerRef) -> TrainerConfig:
        return await self._workflow_builder.get_trainer_config(trainer_name)

    @override
    async def get_trainer_adapter_config(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapterConfig:
        return await self._workflow_builder.get_trainer_adapter_config(trainer_adapter_name)

    @override
    async def get_trajectory_builder_config(
            self, trajectory_builder_name: str | TrajectoryBuilderRef) -> (TrajectoryBuilderConfig):
        return await self._workflow_builder.get_trajectory_builder_config(trajectory_builder_name)

    @override
    async def get_trainer_adapter(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapter:
        return await self._workflow_builder.get_trainer_adapter(trainer_adapter_name)

    @override
    async def get_trajectory_builder(self, trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilder:
        return await self._workflow_builder.get_trajectory_builder(trajectory_builder_name)

    @override
    @experimental(feature_name="TTC")
    async def add_ttc_strategy(self, name: str, config: TTCStrategyBaseConfig) -> None:
        await self._workflow_builder.add_ttc_strategy(name, config)

    @override
    async def get_ttc_strategy(self,
                               strategy_name: str | TTCStrategyRef,
                               pipeline_type: PipelineTypeEnum,
                               stage_type: StageTypeEnum) -> StrategyBase:
        return await self._workflow_builder.get_ttc_strategy(strategy_name=strategy_name,
                                                             pipeline_type=pipeline_type,
                                                             stage_type=stage_type)

    @override
    async def get_ttc_strategy_config(self,
                                      strategy_name: str | TTCStrategyRef,
                                      pipeline_type: PipelineTypeEnum,
                                      stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        return await self._workflow_builder.get_ttc_strategy_config(strategy_name=strategy_name,
                                                                    pipeline_type=pipeline_type,
                                                                    stage_type=stage_type)

    @override
    async def add_retriever(self, name: str, config: RetrieverBaseConfig) -> None:
        await self._workflow_builder.add_retriever(name, config)

    @override
    async def get_retriever(self, retriever_name: str, wrapper_type: LLMFrameworkEnum | str | None = None) -> Retriever:
        if not wrapper_type:
            return await self._workflow_builder.get_retriever(retriever_name=retriever_name)
        return await self._workflow_builder.get_retriever(retriever_name=retriever_name, wrapper_type=wrapper_type)

    @override
    async def get_retriever_config(self, retriever_name: str) -> RetrieverBaseConfig:
        return await self._workflow_builder.get_retriever_config(retriever_name=retriever_name)

    @override
    def get_user_manager(self) -> UserManagerHolder:
        return self._workflow_builder.get_user_manager()

    @override
    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._workflow_builder.get_function_dependencies(fn_name)

    @override
    def get_function_group_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._workflow_builder.get_function_group_dependencies(fn_name)

    @override
    async def add_middleware(self, name: str | MiddlewareRef, config: MiddlewareBaseConfig) -> Middleware:
        """Add middleware to the builder."""
        return await self._workflow_builder.add_middleware(name, config)

    @override
    async def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        """Get built middleware by name."""
        return await self._workflow_builder.get_middleware(middleware_name)

    @override
    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        """Get the configuration for middleware."""
        return self._workflow_builder.get_middleware_config(middleware_name)
