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
"""Synchronous wrapper for accessing Builder instances."""

import asyncio
import typing
from collections.abc import Sequence

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


class SyncBuilder:
    """Synchronous wrapper for the Builder class.

    Provides synchronous access to Builder methods by wrapping async calls with run_until_complete.
    """

    def __init__(self, builder: Builder) -> None:
        self._builder = builder

        try:
            # Save the current loop. This should always be available given the creation pattern of the Builder class.
            self._loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise ValueError("No event loop is running. If you are running the code in a synchronous context, "
                             "please use the async builder instead.") from e

    @staticmethod
    def current() -> "SyncBuilder":
        """Get the SyncBuilder object from the current context.

        Returns:
            The SyncBuilder object wrapping the current Builder, or raises ValueError if not set.
        """
        return SyncBuilder(Builder.current())

    @property
    def async_builder(self) -> Builder:
        """Get the async version of the builder.

        Returns:
            The Builder object (async).
        """
        return self._builder

    def get_function(self, name: str | FunctionRef) -> Function:
        """Get a function by name.

        Args:
            name: The name or reference of the function

        Returns:
            The built function instance
        """
        return self._loop.run_until_complete(self._builder.get_function(name))

    def get_function_group(self, name: str | FunctionGroupRef) -> FunctionGroup:
        """Get a function group by name.

        Args:
            name: The name or reference of the function group

        Returns:
            The built function group instance
        """
        return self._loop.run_until_complete(self._builder.get_function_group(name))

    def get_functions(self, function_names: Sequence[str | FunctionRef]) -> list[Function]:
        """Get multiple functions by name.

        Args:
            function_names: The names or references of the functions

        Returns:
            List of built function instances
        """
        return self._loop.run_until_complete(self._builder.get_functions(function_names))

    def get_function_groups(self, function_group_names: Sequence[str | FunctionGroupRef]) -> list[FunctionGroup]:
        """Get multiple function groups by name.

        Args:
            function_group_names: The names or references of the function groups

        Returns:
            List of built function group instances
        """
        return self._loop.run_until_complete(self._builder.get_function_groups(function_group_names))

    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        """Get the configuration for a function.

        Args:
            name: The name or reference of the function

        Returns:
            The configuration for the function
        """
        return self._builder.get_function_config(name)

    def get_function_group_config(self, name: str | FunctionGroupRef) -> FunctionGroupBaseConfig:
        """Get the configuration for a function group.

        Args:
            name: The name or reference of the function group

        Returns:
            The configuration for the function group
        """
        return self._builder.get_function_group_config(name)

    def get_workflow(self) -> Function:
        """Get the workflow function.

        Returns:
            The workflow function instance
        """
        return self._builder.get_workflow()

    def get_workflow_config(self) -> FunctionBaseConfig:
        """Get the configuration for the workflow.

        Returns:
            The configuration for the workflow function
        """
        return self._builder.get_workflow_config()

    def get_tools(self,
                  tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                  wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get multiple tools by name wrapped in the specified framework type.

        Args:
            tool_names: The names or references of the tools (functions or function groups)
            wrapper_type: The LLM framework type to wrap the tools in

        Returns:
            List of tools wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_tools(tool_names, wrapper_type))

    def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        """Get a tool by name wrapped in the specified framework type.

        Args:
            fn_name: The name or reference of the tool (function)
            wrapper_type: The LLM framework type to wrap the tool in

        Returns:
            The tool wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_tool(fn_name, wrapper_type))

    def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        """Get an LLM by name wrapped in the specified framework type.

        Args:
            llm_name: The name or reference of the LLM
            wrapper_type: The LLM framework type to wrap the LLM in

        Returns:
            The LLM wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_llm(llm_name, wrapper_type))

    def get_llms(self, llm_names: Sequence[str | LLMRef], wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get multiple LLMs by name wrapped in the specified framework type.

        Args:
            llm_names: The names or references of the LLMs
            wrapper_type: The LLM framework type to wrap the LLMs in

        Returns:
            List of LLMs wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_llms(llm_names, wrapper_type))

    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:
        """Get the configuration for an LLM.

        Args:
            llm_name: The name or reference of the LLM

        Returns:
            The configuration for the LLM
        """
        return self._builder.get_llm_config(llm_name)

    def get_auth_provider(self, auth_provider_name: str | AuthenticationRef) -> AuthProviderBase:
        """Get an authentication provider by name.

        Args:
            auth_provider_name: The name or reference of the authentication provider

        Returns:
            The authentication provider instance
        """
        return self._loop.run_until_complete(self._builder.get_auth_provider(auth_provider_name))

    def get_auth_providers(self, auth_provider_names: list[str | AuthenticationRef]) -> list[AuthProviderBase]:
        """Get multiple authentication providers by name.

        Args:
            auth_provider_names: The names or references of the authentication providers

        Returns:
            List of authentication provider instances
        """
        return self._loop.run_until_complete(self._builder.get_auth_providers(auth_provider_names))

    def get_object_store_clients(self, object_store_names: Sequence[str | ObjectStoreRef]) -> list[ObjectStore]:
        """
        Return a list of all object store clients.
        """
        return self._loop.run_until_complete(self._builder.get_object_store_clients(object_store_names))

    def get_object_store_client(self, object_store_name: str | ObjectStoreRef) -> ObjectStore:
        """Get an object store client by name.

        Args:
            object_store_name: The name or reference of the object store

        Returns:
            The object store client instance
        """
        return self._loop.run_until_complete(self._builder.get_object_store_client(object_store_name))

    def get_object_store_config(self, object_store_name: str | ObjectStoreRef) -> ObjectStoreBaseConfig:
        """Get the configuration for an object store.

        Args:
            object_store_name: The name or reference of the object store

        Returns:
            The configuration for the object store
        """
        return self._builder.get_object_store_config(object_store_name)

    def get_embedders(self, embedder_names: Sequence[str | EmbedderRef],
                      wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get multiple embedders by name wrapped in the specified framework type.

        Args:
            embedder_names: The names or references of the embedders
            wrapper_type: The LLM framework type to wrap the embedders in

        Returns:
            List of embedders wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_embedders(embedder_names, wrapper_type))

    def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        """Get an embedder by name wrapped in the specified framework type.

        Args:
            embedder_name: The name or reference of the embedder
            wrapper_type: The LLM framework type to wrap the embedder in

        Returns:
            The embedder wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_embedder(embedder_name, wrapper_type))

    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:
        """Get the configuration for an embedder.

        Args:
            embedder_name: The name or reference of the embedder

        Returns:
            The configuration for the embedder
        """
        return self._builder.get_embedder_config(embedder_name)

    def get_memory_clients(self, memory_names: Sequence[str | MemoryRef]) -> list[MemoryEditor]:
        """
        Return a list of memory clients for the specified names.
        """
        return self._loop.run_until_complete(self._builder.get_memory_clients(memory_names))

    def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        return self._loop.run_until_complete(self._builder.get_memory_client(memory_name))

    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:
        """Get the configuration for a memory client.

        Args:
            memory_name: The name or reference of the memory client

        Returns:
            The configuration for the memory client
        """
        return self._builder.get_memory_client_config(memory_name)

    def get_retrievers(self,
                       retriever_names: Sequence[str | RetrieverRef],
                       wrapper_type: LLMFrameworkEnum | str | None = None) -> list[Retriever]:
        """Get multiple retrievers by name.

        Args:
            retriever_names: The names or references of the retrievers
            wrapper_type: Optional LLM framework type to wrap the retrievers in

        Returns:
            List of retriever instances
        """
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
        """Get a retriever by name.

        Args:
            retriever_name: The name or reference of the retriever
            wrapper_type: Optional LLM framework type to wrap the retriever in

        Returns:
            The retriever instance, optionally wrapped in the specified framework type
        """
        return self._loop.run_until_complete(self._builder.get_retriever(retriever_name, wrapper_type))

    def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:
        """Get the configuration for a retriever.

        Args:
            retriever_name: The name or reference of the retriever

        Returns:
            The configuration for the retriever
        """
        return self._loop.run_until_complete(self._builder.get_retriever_config(retriever_name))

    def get_trainer(self,
                    trainer_name: str | TrainerRef,
                    trajectory_builder: TrajectoryBuilder,
                    trainer_adapter: TrainerAdapter) -> Trainer:
        """Get a trainer by name with the specified trajectory builder and trainer adapter.

        Args:
            trainer_name: The name or reference of the trainer
            trajectory_builder: The trajectory builder instance
            trainer_adapter: The trainer adapter instance

        Returns:
            The trainer instance
        """
        return self._loop.run_until_complete(
            self._builder.get_trainer(trainer_name, trajectory_builder, trainer_adapter))

    def get_trainer_adapter(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapter:
        """Get a trainer adapter by name.

        Args:
            trainer_adapter_name: The name or reference of the trainer adapter

        Returns:
            The trainer adapter instance
        """
        return self._loop.run_until_complete(self._builder.get_trainer_adapter(trainer_adapter_name))

    def get_trajectory_builder(self, trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilder:
        """Get a trajectory builder by name.

        Args:
            trajectory_builder_name: The name or reference of the trajectory builder

        Returns:
            The trajectory builder instance
        """
        return self._loop.run_until_complete(self._builder.get_trajectory_builder(trajectory_builder_name))

    def get_trainer_config(self, trainer_name: str | TrainerRef) -> TrainerConfig:
        """Get the configuration for a trainer.

        Args:
            trainer_name: The name or reference of the trainer

        Returns:
            The configuration for the trainer
        """
        return self._loop.run_until_complete(self._builder.get_trainer_config(trainer_name))

    def get_trainer_adapter_config(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapterConfig:
        """Get the configuration for a trainer adapter.

        Args:
            trainer_adapter_name: The name or reference of the trainer adapter

        Returns:
            The configuration for the trainer adapter
        """
        return self._loop.run_until_complete(self._builder.get_trainer_adapter_config(trainer_adapter_name))

    def get_trajectory_builder_config(self,
                                      trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilderConfig:
        """Get the configuration for a trajectory builder.

        Args:
            trajectory_builder_name: The name or reference of the trajectory builder

        Returns:
            The configuration for the trajectory builder
        """
        return self._loop.run_until_complete(self._builder.get_trajectory_builder_config(trajectory_builder_name))

    def get_ttc_strategy(self,
                         strategy_name: str | TTCStrategyRef,
                         pipeline_type: PipelineTypeEnum,
                         stage_type: StageTypeEnum) -> "StrategyBase":
        """Get a test-time compute strategy by name.

        Args:
            strategy_name: The name or reference of the TTC strategy
            pipeline_type: The pipeline type for the strategy
            stage_type: The stage type for the strategy

        Returns:
            The TTC strategy instance
        """
        return self._loop.run_until_complete(self._builder.get_ttc_strategy(strategy_name, pipeline_type, stage_type))

    def get_ttc_strategy_config(self,
                                strategy_name: str | TTCStrategyRef,
                                pipeline_type: PipelineTypeEnum,
                                stage_type: StageTypeEnum) -> TTCStrategyBaseConfig:
        """Get the configuration for a test-time compute strategy.

        Args:
            strategy_name: The name or reference of the TTC strategy
            pipeline_type: The pipeline type for the strategy
            stage_type: The stage type for the strategy

        Returns:
            The configuration for the TTC strategy
        """
        return self._loop.run_until_complete(
            self._builder.get_ttc_strategy_config(strategy_name, pipeline_type, stage_type))

    def get_user_manager(self) -> UserManagerHolder:
        """Get the user manager holder.

        Returns:
            The user manager holder instance
        """
        return self._builder.get_user_manager()

    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        """Get the dependencies for a function.

        Args:
            fn_name: The name of the function

        Returns:
            The function dependencies
        """
        return self._builder.get_function_dependencies(fn_name)

    def get_function_group_dependencies(self, fn_name: str) -> FunctionDependencies:
        """Get the dependencies for a function group.

        Args:
            fn_name: The name of the function group

        Returns:
            The function group dependencies
        """
        return self._builder.get_function_group_dependencies(fn_name)

    def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        """Get built middleware by name.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The built middleware instance
        """
        return self._loop.run_until_complete(self._builder.get_middleware(middleware_name))

    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        """Get the configuration for middleware.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The configuration for the middleware
        """
        return self._builder.get_middleware_config(middleware_name)

    def get_middleware_list(self, middleware_names: Sequence[str | MiddlewareRef]) -> list[Middleware]:
        """Get multiple middleware by name.

        Args:
            middleware_names: The names or references of the middleware

        Returns:
            List of built middleware instances
        """
        return self._loop.run_until_complete(self._builder.get_middleware_list(middleware_names))
