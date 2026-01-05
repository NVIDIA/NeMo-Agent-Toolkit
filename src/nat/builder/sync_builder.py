# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Static classes for accessing Builder and Config instances stored in ContextVars."""

import asyncio
import typing
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TypeVar

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.builder import UserManagerHolder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import MiddlewareRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
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
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.finetuning.interfaces.finetuning_runner import Trainer
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder
from nat.memory.interfaces import MemoryEditor
from nat.middleware.middleware import Middleware
from nat.object_store.interfaces import ObjectStore
from nat.retriever.interface import Retriever

if typing.TYPE_CHECKING:
    from nat.experimental.test_time_compute.models.strategy_base import StrategyBase

_config_context: ContextVar[FunctionBaseConfig | None] = ContextVar("config", default=None)

T = TypeVar("T", bound=FunctionBaseConfig)


class SyncBuilder:

    def __init__(self, builder: Builder) -> None:
        self._builder = builder

        # Save the current loop
        self._loop = asyncio.get_running_loop()

    @staticmethod
    def current() -> "SyncBuilder":
        return SyncBuilder(Builder.current())

    @property
    def async_builder(self) -> Builder:
        return self._builder

    def get_function(self, name: str | FunctionRef) -> Function:
        return self._loop.run_until_complete(self._builder.get_function(name))

    def get_function_group(self, name: str | FunctionGroupRef) -> FunctionGroup:
        return self._loop.run_until_complete(self._builder.get_function_group(name))

    def get_functions(self, function_names: Sequence[str | FunctionRef]) -> list[Function]:
        return self._loop.run_until_complete(self._builder.get_functions(function_names))

    def get_function_groups(self, function_group_names: Sequence[str | FunctionGroupRef]) -> list[FunctionGroup]:
        return self._loop.run_until_complete(self._builder.get_function_groups(function_group_names))

    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        return self._builder.get_function_config(name)

    def get_function_group_config(self, name: str | FunctionGroupRef) -> FunctionGroupBaseConfig:
        return self._builder.get_function_group_config(name)

    def get_workflow(self) -> Function:
        return self._builder.get_workflow()

    def get_workflow_config(self) -> FunctionBaseConfig:
        return self._builder.get_workflow_config()

    def get_tools(self,
                  tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                  wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        return self._loop.run_until_complete(self._builder.get_tools(tool_names, wrapper_type))

    def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        return self._loop.run_until_complete(self._builder.get_tool(fn_name, wrapper_type))

    def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        return self._loop.run_until_complete(self._builder.get_llm(llm_name, wrapper_type))

    def get_llms(self, llm_names: Sequence[str | LLMRef], wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        return self._loop.run_until_complete(self._builder.get_llms(llm_names, wrapper_type))

    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:
        return self._builder.get_llm_config(llm_name)

    def get_auth_provider(self, auth_provider_name: str | AuthenticationRef) -> AuthProviderBase:
        return self._loop.run_until_complete(self._builder.get_auth_provider(auth_provider_name))

    def get_auth_providers(self, auth_provider_names: list[str | AuthenticationRef]) -> list[AuthProviderBase]:
        return self._loop.run_until_complete(self._builder.get_auth_providers(auth_provider_names))

    def get_object_store_clients(self, object_store_names: Sequence[str | ObjectStoreRef]) -> list[ObjectStore]:
        return self._loop.run_until_complete(self._builder.get_object_store_clients(object_store_names))

    def get_object_store_client(self, object_store_name: str | ObjectStoreRef) -> ObjectStore:
        return self._loop.run_until_complete(self._builder.get_object_store_client(object_store_name))

    def get_object_store_config(self, object_store_name: str | ObjectStoreRef) -> ObjectStoreBaseConfig:
        return self._builder.get_object_store_config(object_store_name)

    def get_embedders(self, embedder_names: Sequence[str | EmbedderRef],
                      wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        return self._loop.run_until_complete(self._builder.get_embedders(embedder_names, wrapper_type))

    def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        return self._loop.run_until_complete(self._builder.get_embedder(embedder_name, wrapper_type))

    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:
        return self._builder.get_embedder_config(embedder_name)

    def get_memory_clients(self, memory_names: Sequence[str | MemoryRef]) -> list[MemoryEditor]:
        return self._loop.run_until_complete(self._builder.get_memory_clients(memory_names))

    def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        return self._loop.run_until_complete(self._builder.get_memory_client(memory_name))

    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:
        return self._builder.get_memory_client_config(memory_name)

    def get_retrievers(self,
                       retriever_names: Sequence[str | RetrieverRef],
                       wrapper_type: LLMFrameworkEnum | str | None = None) -> list[Retriever]:
        return self._loop.run_until_complete(self._builder.get_retrievers(retriever_names, wrapper_type))

    @typing.overload
    def get_retriever(self, retriever_name: str | RetrieverRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        ...

    @typing.overload
    def get_retriever(self, retriever_name: str | RetrieverRef, wrapper_type: None) -> Retriever:
        ...

    @typing.overload
    def get_retriever(self, retriever_name: str | RetrieverRef) -> Retriever:
        ...

    def get_retriever(self,
                      retriever_name: str | RetrieverRef,
                      wrapper_type: LLMFrameworkEnum | str | None = None) -> typing.Any:
        return self._loop.run_until_complete(self._builder.get_retriever(retriever_name, wrapper_type))

    def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:
        return self._loop.run_until_complete(self._builder.get_retriever_config(retriever_name))

    def get_trainer(self,
                    trainer_name: str | TrainerRef,
                    trajectory_builder: TrajectoryBuilder,
                    trainer_adapter: TrainerAdapter) -> Trainer:
        return self._loop.run_until_complete(
            self._builder.get_trainer(trainer_name, trajectory_builder, trainer_adapter))

    def get_trainer_adapter(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapter:
        return self._loop.run_until_complete(self._builder.get_trainer_adapter(trainer_adapter_name))

    def get_trajectory_builder(self, trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilder:
        return self._loop.run_until_complete(self._builder.get_trajectory_builder(trajectory_builder_name))

    def get_trainer_config(self, trainer_name: str | TrainerRef) -> TrainerConfig:
        return self._loop.run_until_complete(self._builder.get_trainer_config(trainer_name))

    def get_trainer_adapter_config(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapterConfig:
        return self._loop.run_until_complete(self._builder.get_trainer_adapter_config(trainer_adapter_name))

    def get_trajectory_builder_config(self,
                                      trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilderConfig:
        return self._loop.run_until_complete(self._builder.get_trajectory_builder_config(trajectory_builder_name))

    def get_ttc_strategy(self,
                         strategy_name: str | TTCStrategyRef,
                         pipeline_type: PipelineTypeEnum,
                         stage_type: StageTypeEnum) -> "StrategyBase":
        return self._loop.run_until_complete(self._builder.get_ttc_strategy(strategy_name, pipeline_type, stage_type))

    def get_ttc_strategy_config(self,
                                strategy_name: str | TTCStrategyRef,
                                pipeline_type: PipelineTypeEnum,
                                stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        return self._loop.run_until_complete(
            self._builder.get_ttc_strategy_config(strategy_name, pipeline_type, stage_type))

    def get_user_manager(self) -> UserManagerHolder:
        return self._builder.get_user_manager()

    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._builder.get_function_dependencies(fn_name)

    def get_function_group_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._builder.get_function_group_dependencies(fn_name)

    def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        return self._loop.run_until_complete(self._builder.get_middleware(middleware_name))

    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        return self._builder.get_middleware_config(middleware_name)

    def get_middleware_list(self, middleware_names: Sequence[str | MiddlewareRef]) -> list[Middleware]:
        return self._loop.run_until_complete(self._builder.get_middleware_list(middleware_names))
