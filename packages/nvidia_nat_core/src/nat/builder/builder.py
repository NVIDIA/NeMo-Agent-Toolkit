# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.data_models.authentication import AuthProviderBaseConfig
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
from nat.data_models.evaluator import EvaluatorBaseConfig
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
from nat.finetuning.interfaces.finetuning_runner import Trainer
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder
from nat.memory.interfaces import MemoryEditor
from nat.middleware.middleware import Middleware
from nat.object_store.interfaces import ObjectStore
from nat.retriever.interface import Retriever

if typing.TYPE_CHECKING:
    from nat.builder.sync_builder import SyncBuilder
    from nat.experimental.test_time_compute.models.strategy_base import StrategyBase

_current_builder_context: ContextVar["Builder | None"] = ContextVar("current_builder", default=None)


class UserManagerHolder:

    def __init__(self, context: Context) -> None:
        self._context = context

    def get_id(self):
        return self._context.user_manager.get_id()


class Builder(ABC):

    @staticmethod
    def current() -> "Builder":
        """Get the Builder object from the current context.

        Returns:
            The Builder object stored in the ContextVar, or raises ValueError if not set.
        """
        builder = _current_builder_context.get()

        if builder is None:
            raise ValueError("Builder not set in context")

        return builder

    @property
    @abstractmethod
    def sync_builder(self) -> "SyncBuilder":
        """Get the synchronous version of the builder.

        Returns:
            The SyncBuilder object (synchronous wrapper).
        """
        pass

    @abstractmethod
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:
        """Add a function to the builder.

        Args:
            name: The name or reference for the function
            config: The configuration for the function

        Returns:
            The built function instance
        """
        pass

    @abstractmethod
    async def add_function_group(self, name: str | FunctionGroupRef, config: FunctionGroupBaseConfig) -> FunctionGroup:
        """Add a function group to the builder.

        Args:
            name: The name or reference for the function group
            config: The configuration for the function group

        Returns:
            The built function group instance
        """
        pass

    @abstractmethod
    async def get_function(self, name: str | FunctionRef) -> Function:
        """Get a function by name.

        Args:
            name: The name or reference of the function

        Returns:
            The built function instance
        """
        pass

    @abstractmethod
    async def get_function_group(self, name: str | FunctionGroupRef) -> FunctionGroup:
        """Get a function group by name.

        Args:
            name: The name or reference of the function group

        Returns:
            The built function group instance
        """
        pass

    async def get_functions(self, function_names: Sequence[str | FunctionRef]) -> list[Function]:
        """Get multiple functions by name.

        Args:
            function_names: The names or references of the functions

        Returns:
            List of built function instances
        """
        tasks = [self.get_function(name) for name in function_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def get_function_groups(self, function_group_names: Sequence[str | FunctionGroupRef]) -> list[FunctionGroup]:
        """Get multiple function groups by name.

        Args:
            function_group_names: The names or references of the function groups

        Returns:
            List of built function group instances
        """
        tasks = [self.get_function_group(name) for name in function_group_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    @abstractmethod
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        """Get the configuration for a function.

        Args:
            name: The name or reference of the function

        Returns:
            The configuration for the function
        """
        pass

    @abstractmethod
    def get_function_group_config(self, name: str | FunctionGroupRef) -> FunctionGroupBaseConfig:
        """Get the configuration for a function group.

        Args:
            name: The name or reference of the function group

        Returns:
            The configuration for the function group
        """
        pass

    @abstractmethod
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        """Set the workflow function.

        Args:
            config: The configuration for the workflow function

        Returns:
            The built workflow function instance
        """
        pass

    @abstractmethod
    def get_workflow(self) -> Function:
        """Get the workflow function.

        Returns:
            The workflow function instance
        """
        pass

    @abstractmethod
    def get_workflow_config(self) -> FunctionBaseConfig:
        """Get the configuration for the workflow.

        Returns:
            The configuration for the workflow function
        """
        pass

    @abstractmethod
    async def get_tools(self,
                        tool_names: Sequence[str | FunctionRef | FunctionGroupRef],
                        wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get multiple tools by name wrapped in the specified framework type.

        Args:
            tool_names: The names or references of the tools (functions or function groups)
            wrapper_type: The LLM framework type to wrap the tools in

        Returns:
            List of tools wrapped in the specified framework type
        """
        pass

    @abstractmethod
    async def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        """Get a tool by name wrapped in the specified framework type.

        Args:
            fn_name: The name or reference of the tool (function)
            wrapper_type: The LLM framework type to wrap the tool in

        Returns:
            The tool wrapped in the specified framework type
        """
        pass

    @abstractmethod
    async def add_llm(self, name: str | LLMRef, config: LLMBaseConfig) -> typing.Any:
        """Add an LLM to the builder.

        Args:
            name: The name or reference for the LLM
            config: The configuration for the LLM

        Returns:
            The built LLM instance
        """
        pass

    @abstractmethod
    async def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        """Get an LLM by name wrapped in the specified framework type.

        Args:
            llm_name: The name or reference of the LLM
            wrapper_type: The LLM framework type to wrap the LLM in

        Returns:
            The LLM wrapped in the specified framework type
        """
        pass

    async def get_llms(self, llm_names: Sequence[str | LLMRef],
                       wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get multiple LLMs by name wrapped in the specified framework type.

        Args:
            llm_names: The names or references of the LLMs
            wrapper_type: The LLM framework type to wrap the LLMs in

        Returns:
            List of LLMs wrapped in the specified framework type
        """
        coros = [self.get_llm(llm_name=n, wrapper_type=wrapper_type) for n in llm_names]

        llms = await asyncio.gather(*coros, return_exceptions=False)

        return list(llms)

    @abstractmethod
    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:
        """Get the configuration for an LLM.

        Args:
            llm_name: The name or reference of the LLM

        Returns:
            The configuration for the LLM
        """
        pass

    @abstractmethod
    @experimental(feature_name="Authentication")
    async def add_auth_provider(self, name: str | AuthenticationRef,
                                config: AuthProviderBaseConfig) -> AuthProviderBase:
        """Add an authentication provider to the builder.

        Args:
            name: The name or reference for the authentication provider
            config: The configuration for the authentication provider

        Returns:
            The built authentication provider instance
        """
        pass

    @abstractmethod
    async def get_auth_provider(self, auth_provider_name: str | AuthenticationRef) -> AuthProviderBase:
        """Get an authentication provider by name.

        Args:
            auth_provider_name: The name or reference of the authentication provider

        Returns:
            The authentication provider instance
        """
        pass

    async def get_auth_providers(self, auth_provider_names: list[str | AuthenticationRef]):
        """Get multiple authentication providers by name.

        Args:
            auth_provider_names: The names or references of the authentication providers

        Returns:
            List of authentication provider instances
        """
        coros = [self.get_auth_provider(auth_provider_name=n) for n in auth_provider_names]

        auth_providers = await asyncio.gather(*coros, return_exceptions=False)

        return list(auth_providers)

    @abstractmethod
    async def add_object_store(self, name: str | ObjectStoreRef, config: ObjectStoreBaseConfig) -> ObjectStore:
        """Add an object store to the builder.

        Args:
            name: The name or reference for the object store
            config: The configuration for the object store

        Returns:
            The built object store instance
        """
        pass

    async def get_object_store_clients(self, object_store_names: Sequence[str | ObjectStoreRef]) -> list[ObjectStore]:
        """
        Return a list of all object store clients.
        """
        return list(await asyncio.gather(*[self.get_object_store_client(name) for name in object_store_names]))

    @abstractmethod
    async def get_object_store_client(self, object_store_name: str | ObjectStoreRef) -> ObjectStore:
        """Get an object store client by name.

        Args:
            object_store_name: The name or reference of the object store

        Returns:
            The object store client instance
        """
        pass

    @abstractmethod
    def get_object_store_config(self, object_store_name: str | ObjectStoreRef) -> ObjectStoreBaseConfig:
        """Get the configuration for an object store.

        Args:
            object_store_name: The name or reference of the object store

        Returns:
            The configuration for the object store
        """
        pass

    @abstractmethod
    async def add_embedder(self, name: str | EmbedderRef, config: EmbedderBaseConfig) -> None:
        """Add an embedder to the builder.

        Args:
            name: The name or reference for the embedder
            config: The configuration for the embedder
        """
        pass

    async def get_embedders(self, embedder_names: Sequence[str | EmbedderRef],
                            wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get multiple embedders by name wrapped in the specified framework type.

        Args:
            embedder_names: The names or references of the embedders
            wrapper_type: The LLM framework type to wrap the embedders in

        Returns:
            List of embedders wrapped in the specified framework type
        """
        coros = [self.get_embedder(embedder_name=n, wrapper_type=wrapper_type) for n in embedder_names]

        embedders = await asyncio.gather(*coros, return_exceptions=False)

        return list(embedders)

    @abstractmethod
    async def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        """Get an embedder by name wrapped in the specified framework type.

        Args:
            embedder_name: The name or reference of the embedder
            wrapper_type: The LLM framework type to wrap the embedder in

        Returns:
            The embedder wrapped in the specified framework type
        """
        pass

    @abstractmethod
    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:
        """Get the configuration for an embedder.

        Args:
            embedder_name: The name or reference of the embedder

        Returns:
            The configuration for the embedder
        """
        pass

    @abstractmethod
    async def add_memory_client(self, name: str | MemoryRef, config: MemoryBaseConfig) -> MemoryEditor:
        """Add a memory client to the builder.

        Args:
            name: The name or reference for the memory client
            config: The configuration for the memory client

        Returns:
            The built memory client instance
        """
        pass

    async def get_memory_clients(self, memory_names: Sequence[str | MemoryRef]) -> list[MemoryEditor]:
        """
        Return a list of memory clients for the specified names.
        """
        tasks = [self.get_memory_client(n) for n in memory_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    @abstractmethod
    async def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """
        Return the instantiated memory client for the given name.
        """
        pass

    @abstractmethod
    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:
        """Get the configuration for a memory client.

        Args:
            memory_name: The name or reference of the memory client

        Returns:
            The configuration for the memory client
        """
        pass

    @abstractmethod
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig) -> None:
        """Add a retriever to the builder.

        Args:
            name: The name or reference for the retriever
            config: The configuration for the retriever
        """
        pass

    async def get_retrievers(self,
                             retriever_names: Sequence[str | RetrieverRef],
                             wrapper_type: LLMFrameworkEnum | str | None = None) -> list[Retriever]:
        """Get multiple retrievers by name.

        Args:
            retriever_names: The names or references of the retrievers
            wrapper_type: Optional LLM framework type to wrap the retrievers in

        Returns:
            List of retriever instances
        """
        tasks = [self.get_retriever(n, wrapper_type=wrapper_type) for n in retriever_names]

        retrievers = await asyncio.gather(*tasks, return_exceptions=False)

        return list(retrievers)

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str) -> typing.Any:
        ...

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef, wrapper_type: None) -> Retriever:
        ...

    @typing.overload
    async def get_retriever(self, retriever_name: str | RetrieverRef) -> Retriever:
        ...

    @abstractmethod
    async def get_retriever(self,
                            retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str | None = None) -> typing.Any:
        """Get a retriever by name.

        Args:
            retriever_name: The name or reference of the retriever
            wrapper_type: Optional LLM framework type to wrap the retriever in

        Returns:
            The retriever instance, optionally wrapped in the specified framework type
        """
        pass

    @abstractmethod
    async def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:
        """Get the configuration for a retriever.

        Args:
            retriever_name: The name or reference of the retriever

        Returns:
            The configuration for the retriever
        """
        pass

    @abstractmethod
    @experimental(feature_name="Finetuning")
    async def add_trainer(self, name: str | TrainerRef, config: TrainerConfig) -> Trainer:
        """Add a trainer to the builder.

        Args:
            name: The name or reference for the trainer
            config: The configuration for the trainer

        Returns:
            The built trainer instance
        """
        pass

    @abstractmethod
    @experimental(feature_name="Finetuning")
    async def add_trainer_adapter(self, name: str | TrainerAdapterRef, config: TrainerAdapterConfig) -> TrainerAdapter:
        """Add a trainer adapter to the builder.

        Args:
            name: The name or reference for the trainer adapter
            config: The configuration for the trainer adapter

        Returns:
            The built trainer adapter instance
        """
        pass

    @abstractmethod
    @experimental(feature_name="Finetuning")
    async def add_trajectory_builder(self, name: str | TrajectoryBuilderRef,
                                     config: TrajectoryBuilderConfig) -> TrajectoryBuilder:
        """Add a trajectory builder to the builder.

        Args:
            name: The name or reference for the trajectory builder
            config: The configuration for the trajectory builder

        Returns:
            The built trajectory builder instance
        """
        pass

    @abstractmethod
    async def get_trainer(self,
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
        pass

    @abstractmethod
    async def get_trainer_adapter(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapter:
        """Get a trainer adapter by name.

        Args:
            trainer_adapter_name: The name or reference of the trainer adapter

        Returns:
            The trainer adapter instance
        """
        pass

    @abstractmethod
    async def get_trajectory_builder(self, trajectory_builder_name: str | TrajectoryBuilderRef) -> TrajectoryBuilder:
        """Get a trajectory builder by name.

        Args:
            trajectory_builder_name: The name or reference of the trajectory builder

        Returns:
            The trajectory builder instance
        """
        pass

    @abstractmethod
    async def get_trainer_config(self, trainer_name: str | TrainerRef) -> TrainerConfig:
        """Get the configuration for a trainer.

        Args:
            trainer_name: The name or reference of the trainer

        Returns:
            The configuration for the trainer
        """
        pass

    @abstractmethod
    async def get_trainer_adapter_config(self, trainer_adapter_name: str | TrainerAdapterRef) -> TrainerAdapterConfig:
        """Get the configuration for a trainer adapter.

        Args:
            trainer_adapter_name: The name or reference of the trainer adapter

        Returns:
            The configuration for the trainer adapter
        """
        pass

    @abstractmethod
    async def get_trajectory_builder_config(
            self, trajectory_builder_name: str | TrajectoryBuilderRef) -> (TrajectoryBuilderConfig):
        """Get the configuration for a trajectory builder.

        Args:
            trajectory_builder_name: The name or reference of the trajectory builder

        Returns:
            The configuration for the trajectory builder
        """
        pass

    @abstractmethod
    @experimental(feature_name="TTC")
    async def add_ttc_strategy(self, name: str | TTCStrategyRef, config: TTCStrategyBaseConfig):
        """Add a test-time compute strategy to the builder.

        Args:
            name: The name or reference for the TTC strategy
            config: The configuration for the TTC strategy
        """
        pass

    @abstractmethod
    async def get_ttc_strategy(self,
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
        pass

    @abstractmethod
    async def get_ttc_strategy_config(self,
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
        pass

    @abstractmethod
    def get_user_manager(self) -> UserManagerHolder:
        """Get the user manager holder.

        Returns:
            The user manager holder instance
        """
        pass

    @abstractmethod
    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        """Get the dependencies for a function.

        Args:
            fn_name: The name of the function

        Returns:
            The function dependencies
        """
        pass

    @abstractmethod
    def get_function_group_dependencies(self, fn_name: str) -> FunctionDependencies:
        """Get the dependencies for a function group.

        Args:
            fn_name: The name of the function group

        Returns:
            The function group dependencies
        """
        pass

    @abstractmethod
    async def add_middleware(self, name: str | MiddlewareRef, config: MiddlewareBaseConfig) -> Middleware:
        """Add middleware to the builder.

        Args:
            name: The name or reference for the middleware
            config: The configuration for the middleware

        Returns:
            The built middleware instance
        """
        pass

    @abstractmethod
    async def get_middleware(self, middleware_name: str | MiddlewareRef) -> Middleware:
        """Get built middleware by name.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The built middleware instance
        """
        pass

    @abstractmethod
    def get_middleware_config(self, middleware_name: str | MiddlewareRef) -> MiddlewareBaseConfig:
        """Get the configuration for middleware.

        Args:
            middleware_name: The name or reference of the middleware

        Returns:
            The configuration for the middleware
        """
        pass

    async def get_middleware_list(self, middleware_names: Sequence[str | MiddlewareRef]) -> list[Middleware]:
        """Get multiple middleware by name.

        Args:
            middleware_names: The names or references of the middleware

        Returns:
            List of built middleware instances
        """
        tasks = [self.get_middleware(name) for name in middleware_names]
        return list(await asyncio.gather(*tasks, return_exceptions=False))


class EvalBuilder(ABC):
    """Abstract base class for evaluation builder functionality."""

    @abstractmethod
    async def add_evaluator(self, name: str, config: EvaluatorBaseConfig):
        """Add an evaluator to the builder.

        Args:
            name: The name for the evaluator
            config: The configuration for the evaluator
        """
        pass

    @abstractmethod
    def get_evaluator(self, evaluator_name: str) -> typing.Any:
        """Get an evaluator by name.

        Args:
            evaluator_name: The name of the evaluator

        Returns:
            The evaluator instance
        """
        pass

    @abstractmethod
    def get_evaluator_config(self, evaluator_name: str) -> EvaluatorBaseConfig:
        """Get the configuration for an evaluator.

        Args:
            evaluator_name: The name of the evaluator

        Returns:
            The configuration for the evaluator
        """
        pass

    @abstractmethod
    def get_max_concurrency(self) -> int:
        """Get the maximum concurrency for evaluation.

        Returns:
            The maximum concurrency value
        """
        pass

    @abstractmethod
    def get_output_dir(self) -> Path:
        """Get the output directory for evaluation results.

        Returns:
            The output directory path
        """
        pass

    @abstractmethod
    async def get_all_tools(self, wrapper_type: LLMFrameworkEnum | str) -> list[typing.Any]:
        """Get all tools wrapped in the specified framework type.

        Args:
            wrapper_type: The LLM framework type to wrap the tools in

        Returns:
            List of all tools wrapped in the specified framework type
        """
        pass
